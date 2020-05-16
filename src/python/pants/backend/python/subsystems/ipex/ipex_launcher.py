# Copyright 2020 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Entrypoint script for a "dehydrated" .ipex file generated with --generate-ipex.

This script will "hydrate" a normal .pex file in the same directory, then execute it.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from urllib.request import urlopen
from pkg_resources import Distribution, Requirement

from pex import third_party
from pex.pex_bootstrapper import bootstrap_pex
from pex.common import atomic_directory, open_zip
from pex.interpreter import PythonInterpreter
from pex.jobs import Job, SpawnedJob, execute_parallel
from pex.pex_builder import PEXBuilder
from pex.pex_info import PexInfo
from pex.resolver import resolve
from pex.third_party import isolated
from pex.tracer import TRACER
from pex.variables import ENV


APP_CODE_PREFIX = "user_files/"


def _strip_app_code_prefix(path):
    if not path.startswith(APP_CODE_PREFIX):
        raise ValueError(
            "Path {path} in IPEX-INFO did not begin with '{APP_CODE_PREFIX}'.".format(
                path=path, APP_CODE_PREFIX=APP_CODE_PREFIX
            )
        )
    return path[len(APP_CODE_PREFIX):]


# TODO: replace _log() and _timed() with just TRACER.log() and TRACER.timed() -- it's not clear how
# to make sure messages are printed to stderr (an env var or something may be necessary?)!
def _log(message):
    TRACER.log(message, V=3)
    sys.stderr.write(message + "\n")


@contextmanager
def _timed(message):
    _log(message)
    with TRACER.timed(message, V=3):
        yield


def modify_pex_info(pex_info, **kwargs):
    new_info = json.loads(pex_info.dump())
    new_info.update(kwargs)
    return PexInfo.from_json(json.dumps(new_info))


def _launch_thread(spawn_func, name, tracing_span_description):
    def wrapper():
        try:
            with _timed(tracing_span_description):
                spawn_func()
        except Exception:
            # TODO: add testing for this branch!
            # Print the exception text with full traceback to stderr immediately upon any error.
            import traceback
            traceback.print_exc()
            # FIXME: Right now, we want to immediately die loudly whenever any error occurs, the
            # same as if we were running this all in a single thread. Sending the kill signal to the
            # current process appears to work reliably for that purpose.
            import signal
            os.kill(os.getpid(), signal.SIGKILL)

    t = threading.Thread(target=wrapper, name=name)
    return t


def _wait_on_threads_save_cwd(threads_to_join):
    # FIXME: figure out how to pass an arbitrary cwd to PEX.run() without having to mutate
    # process-global state with os.chdir()!
    prev_cwd = os.getcwd()

    try:
        for t in threads_to_join:
            t.join()
    finally:
        os.chdir(prev_cwd)


class SdistInstaller(object):
    @classmethod
    def create(cls, path):
        """Creates an installer tool with PEX isolation at path.

        :param str path: The path to build the sdist installer tool pex at.
        """
        sdist_installer_pex_path = os.path.join(path, isolated().pex_hash)
        with atomic_directory(sdist_installer_pex_path) as chroot:
            if chroot is not None:
                from pex.pex_builder import PEXBuilder

                isolated_sdist_installer_builder = PEXBuilder(path=chroot)

                # FIXME: the vendored 'setuptools' and 'wheel' don't appear to contain direct import
                # access to a 'pyparsing' dist, which is expected by setup.py files in the
                # wild. Resolving these packages from pypi instead (just once, as this pex's
                # creation is cached) gives us the expected transitive deps.
                for resolved_dist in resolve(['setuptools', 'wheel']):
                    cur_dist = resolved_dist.distribution
                    cur_req = cur_dist.as_requirement()
                    isolated_sdist_installer_builder.add_distribution(dist=cur_dist)
                    isolated_sdist_installer_builder.add_requirement(cur_req)

                isolated_sdist_installer_builder.freeze()

        return cls(sdist_installer_pex_path)

    def __init__(self, sdist_installer_pex_path):
        self._sdist_installer_pex_path = sdist_installer_pex_path

    _INSTALL_PREFIX_SUBDIR = 'install_prefix_subdir'

    def install_sdist(self, dist_base, cache=None, interpreter=None):
        install_dir = os.path.join(dist_base, self._INSTALL_PREFIX_SUBDIR)
        assert os.path.isabs(install_dir), install_dir

        command = ['setup.py', 'install', '--prefix', install_dir]
        # FIXME: --no-compile *should* be speeding these up, but it instead causes the absl-py
        # .tar.gz sdist to create a zipped .egg file on 'install --no-compile', which causes pex to
        # fail to recognize the dist when we try to add it to the hydrated pex we are building.
        # command = ['setup.py', 'install', '--prefix', install_dir, '--no-compile']

        # TODO: most of the below environment manipulation to execute the pex file is cargo-culted
        # from the similar pip pex tool in pex/pip.py in the pex repo!!!
        pex_verbosity = ENV.PEX_VERBOSE
        from pex.pex import PEX
        sdist_installer = PEX(pex=self._sdist_installer_pex_path, interpreter=interpreter)

        env = dict(
            PEX_ROOT=cache or ENV.PEX_ROOT,
            PEX_VERBOSE=str(pex_verbosity),
            # This allows us to import files adjacent to setup.py, which is expected by setup.py
            # files in the wild. Note that we restrict the PYTHONPATH to just the directory
            # containing the setup.py file (and the install directory).
            PEX_INHERIT_PATH='fallback',
            PYTHONPATH=os.pathsep.join([
                dist_base,
                # This silences a warning that appears when the install dir is not contained in the
                # PYTHONPATH.
                install_dir,
            ]),
        )
        with ENV.strip().patch(**env) as env:
            # FIXME: It's not possible to pass in an arbitrary cwd to PEX.run() without setting the
            # global process cwd. If we make no assumptions about the state of the cwd afterwards,
            # we should (hopefully!) be able to safely set it here and expect it to be read without
            # being pre-empted by another thread in between.
            os.chdir(dist_base)
            sdist_installer.run(
                args=command,
                blocking=True,
                env=env,
                with_chroot=False,
            )

        globbed_path_entry = glob.glob(os.path.join(install_dir,
                                                    'lib/python*/site-packages'))
        assert len(globbed_path_entry) == 1, 'globbed_path_entry was {}'.format(globbed_path_entry)
        _log('manually-installed non-wheel dist {}'.format(globbed_path_entry[0]))
        return os.path.join(dist_base, globbed_path_entry[0])


_SDIST_INSTALLER = None


def get_sdist_installer():
  """Returns a lazily instantiated global sdist installer object that is safe for un-coordinated use."""
  global _SDIST_INSTALLER
  if _SDIST_INSTALLER is None:
    _SDIST_INSTALLER = SdistInstaller.create(path=os.path.join(ENV.PEX_ROOT, 'sdist-installer.pex'))
  return _SDIST_INSTALLER


# FIXME: this method should be using some more official algorithm to parse these URLs!
def _extract_download_filename(name, url):
    matched = re.match(r'^https?://.*/([^/]+)\.(whl|WHL|tar\.gz)#sha256=.*$', url)
    if not matched:
        raise TypeError('url for project {} did not match expected format: {}'.format(name, url))
    filename_base, ext = matched.groups()
    download_filename = '{}.{}'.format(filename_base, ext)
    return download_filename


class _RemoteDistribution(object):
    """A description of where and how to download a remote distribution.

    This will play nicely with execute_parallel() and will implement some caching logic.
    """

    def __init__(self, name, version, url, filename):
        self.name = name
        self.version = version
        self.url = url
        self.filename = filename

    def download_maybe_cached_maybe_extract(self, output_dir):
        raise NotImplemented


class _RemoteWheel(_RemoteDistribution):

    def __init__(self, name, version, url, filename):
        assert filename.endswith('.whl'), 'wheel should end in .whl, but was: {}'.format(filename)
        super(_RemoteWheel, self).__init__(name=name, version=version, url=url, filename=filename)

    def download_maybe_cached_maybe_extract(self, output_dir):
        wheel_name = os.path.basename(self.filename)
        downloaded_subdir_abspath = '{}/{}'.format(
            output_dir,
            re.sub(r'\.whl$', '.whl-downloaded', self.filename))
        wheel_output_path = '{}/{}'.format(
            output_dir,
            wheel_name,
        )
        if os.path.exists(wheel_output_path):
            return wheel_output_path

        with atomic_directory(downloaded_subdir_abspath) as containing_dir:
            if containing_dir:
                download_file_abspath = '{}/{}'.format(containing_dir, wheel_name)
                with _timed('extracting wheel from {} into {}'
                            .format(self.url, download_file_abspath)):
                    with urlopen(self.url) as response,\
                         open(download_file_abspath, 'wb') as download_file_stream:
                        shutil.copyfileobj(response, download_file_stream)

        download_file_abspath = '{}/{}'.format(downloaded_subdir_abspath, wheel_name)
        os.symlink(src=download_file_abspath,
                   dst=wheel_output_path)

        assert os.path.exists(wheel_output_path), wheel_output_path
        return wheel_output_path


class _RemoteTarGz(_RemoteDistribution):

    def __init__(self, name, version, url, filename):
        assert filename.endswith('.tar.gz'), 'non-wheel must be a .tar.gz, but was: {filename}'.format(filename=filename)
        filename = re.sub(r'\.tar\.gz$', '', filename)
        super(_RemoteTarGz, self).__init__(name=name, version=version, url=url, filename=filename)

    @staticmethod
    def _sdist_path_from_extracted_archive(download_path):
        # TODO: We currently extract .tar.gz files into a subdirectory named the same as the parent
        # directory. This might be possible to avoid.
        return os.path.join(download_path, os.path.basename(download_path))

    def download_maybe_cached_maybe_extract(self, output_dir):
        final_download_path = os.path.join(output_dir, self.filename)
        with atomic_directory(final_download_path) as download_path:
            if download_path:
                with _timed('extracting .tar.gz archive from {} into {}'
                            .format(self.url, download_path)):
                    with urlopen(self.url) as response,\
                         tarfile.open(mode='r|gz', fileobj=response) as archive_stream:
                        archive_stream.extractall(path=download_path)
        return self._sdist_path_from_extracted_archive(final_download_path)


def _download_urls_parallel(output_dir, remote_distributions):
    pool = ThreadPool(processes=len(remote_distributions))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    def download_dist_from_url(remote_dist):
        with _timed(
                'Downloading and/ or extracting {} into directory {}'
                .format(remote_dist.url, output_dir)):
            extracted_path = remote_dist.download_maybe_cached_maybe_extract(output_dir)
            _log('downloaded and/or extracted {} into {}'.format(remote_dist.url, extracted_path))

        return (remote_dist.name, remote_dist.version, extracted_path)

    try:
        yield from pool.imap_unordered(download_dist_from_url, remote_distributions)
    finally:
        pool.close()


def _resolve_non_wheels(non_wheel_output_dir, non_wheels, bootstrap_builder):
    _log('non_wheel_output_dir: {}'.format(non_wheel_output_dir))
    _log('non_wheels: {}'.format(non_wheels))
    # Note that this iterator should *not* block on the urls downloading yet!
    all_non_wheel_requirements = _download_urls_parallel(
        output_dir=non_wheel_output_dir,
        remote_distributions=non_wheels)

    def install_sdist(args):
        name, version, sdist_path = tuple(args)

        with _timed('installing sdist at {}'.format(sdist_path)):
            site_packages_path = get_sdist_installer().install_sdist(sdist_path)

        assert os.path.isdir(site_packages_path), site_packages_path
        return (name, version, site_packages_path)

    pool = ThreadPool(processes=len(non_wheels))

    try:
        for name, version, site_packages_path in pool.imap_unordered(
                install_sdist,
                all_non_wheel_requirements):
            with _timed('adding installed sdist at {}'.format(site_packages_path)):
                bootstrap_builder.add_dist_location(site_packages_path, name=name)
                bootstrap_builder.add_requirement('{name}=={version}'
                                                  .format(name=name, version=version))
    finally:
        pool.close()


def _resolve_wheels(wheel_output_dir, wheels, bootstrap_builder):
    _log('wheel_output_dir: {}'.format(wheel_output_dir))
    _log('wheels: {}'.format(wheels))
    for name, version, download_path in _download_urls_parallel(
            output_dir=wheel_output_dir,
            remote_distributions=wheels):
        with _timed('adding wheel distribution {}'.format(download_path)):
            bootstrap_builder.add_dist_location(download_path, name=name)
            bootstrap_builder.add_requirement('{name}=={version}'
                                              .format(name=name, version=version))


class _HydrateableRequirementSet(object):
    """Encapsulate all the cacheable inputs required for hydration."""

    @classmethod
    def create(cls, requirements_with_urls, bootstrap_builder):
        wheels = []
        non_wheels = []
        for requirement_string, url in requirements_with_urls.items():
            requirement = Requirement.parse(requirement_string)
            name = requirement.name
            specs = requirement.specs
            assert ((len(specs) == 1) and specs[0][0] == '=='), 'requirement specs are expected to be a single == relation: was {specs}'.format(specs=specs)
            version = specs[0][1]
            download_filename = _extract_download_filename(name, url)
            if download_filename.endswith('.whl'):
                remote_wheel = _RemoteWheel(name=name, version=version, url=url, filename=download_filename)
                wheels.append(remote_wheel)
            else:
                remote_non_wheel = _RemoteTarGz(name=name, version=version, url=url, filename=download_filename)
                non_wheels.append(remote_non_wheel)

        return cls(
            remote_wheel_requests=wheels,
            remote_tar_gz_requests=non_wheels,
            bootstrap_builder=bootstrap_builder,
        )

    def __init__(self, remote_wheel_requests, remote_tar_gz_requests, bootstrap_builder):
        self._remote_wheel_requests = remote_wheel_requests
        self._remote_tar_gz_requests = remote_tar_gz_requests
        self._bootstrap_builder = bootstrap_builder

    def perform_hacky_pinned_resolve(self):
        bootstrap_info = self._bootstrap_builder.info
        ipex_downloads_cache = os.path.join(bootstrap_info.pex_root, 'ipex-downloads')

        non_wheel_output_dir = os.path.join(ipex_downloads_cache, 'non-wheel')
        non_wheels_task = _launch_thread(
            lambda: _resolve_non_wheels(non_wheel_output_dir, self._remote_tar_gz_requests,
                                        self._bootstrap_builder),
            name='fetch-non-wheels',
            tracing_span_description='add non-wheel (wheelified) distributions')

        wheel_output_dir = os.path.join(ipex_downloads_cache, 'wheel')
        wheels_task = _launch_thread(
            lambda: _resolve_wheels(wheel_output_dir, self._remote_wheel_requests,
                                    self._bootstrap_builder),
            name='fetch-wheels',
            tracing_span_description='add wheels')

        all_tasks = [non_wheels_task, wheels_task]
        for t in all_tasks:
            t.start()
        _wait_on_threads_save_cwd(all_tasks)


def _hydrate_pex_file(self, hydrated_pex_dir):
    # We extract source files into a temporary directory before creating the pex.
    td = tempfile.mkdtemp()

    with open_zip(self) as zf:
        # Populate the pex with the pinned requirements and distribution names & hashes.
        bootstrap_info = PexInfo.from_json(zf.read("BOOTSTRAP-PEX-INFO"))
        bootstrap_builder = PEXBuilder(pex_info=bootstrap_info, interpreter=PythonInterpreter.get())

        # Populate the pex with the needed code.
        try:
            ipex_info = json.loads(zf.read("IPEX-INFO").decode("utf-8"))
            for path in ipex_info["code"]:
                unzipped_source = zf.extract(path, td)
                bootstrap_builder.add_source(
                    unzipped_source, env_filename=_strip_app_code_prefix(path)
                )
        except Exception as e:
            raise ValueError(
                "Error: {e}. The IPEX-INFO for this .ipex file was:\n{info}".format(
                    e=e, info=json.dumps(ipex_info, indent=4)
                )
            )

    # Perform a fully pinned intransitive resolve, in parallel directly from requirement URLs.
    requirements_with_urls = ipex_info['requirements_with_urls']

    req_set = _HydrateableRequirementSet.create(requirements_with_urls, bootstrap_builder)
    req_set.perform_hacky_pinned_resolve()

    # TODO: Currently, absl-py will fail to be resolved when the frozen pex dir of the example using
    # tensorflow==1.14.0 is executed! Note that absl-py is provided as an sdist via a .tar.gz, so
    # this means we need to do one more thing somewhere to register these types of dists so that pex
    # won't error at startup.
    bootstrap_builder.info.ignore_errors = True

    # NB: Bytecode compilation can take an extremely long time for large 3rdparty modules.
    bootstrap_builder.freeze(bytecode_compile=False)
    os.rename(src=bootstrap_builder.path(),
              dst=hydrated_pex_dir)


def main(self):
    filename_base, ext = os.path.splitext(self)

    # Incorporate the code hash into the output unpacked pex directory in order to:
    # (a) avoid execing an out of date hydrated ipex,
    # (b) avoid collisions with other similarly-named (i)pex files in the same directory!
    code_hash = PexInfo.from_pex(self).code_hash
    hydrated_pex_dir = "{}-{}{}".format(filename_base, code_hash, ext)

    if not os.path.exists(hydrated_pex_dir):
        with _timed("Hydrating {} to {}...".format(self, hydrated_pex_dir)):
            _hydrate_pex_file(self, hydrated_pex_dir)

    # FIXME: wheel-based dependencies (specifically tensorflow) are not added to the sys.path
    # (possibly related with the reason we set ignore_errors=True above) for some reason!!!
    deps_dirs = glob.glob(os.path.join(hydrated_pex_dir, '.deps/*/'))
    sys.path.extend(deps_dirs)

    # TODO: document this env var!
    if 'IPEX_SKIP_EXECUTION' not in os.environ:
        bootstrap_pex(hydrated_pex_dir)


if __name__ == "__main__":
    self = sys.argv[0]
    main(self)
