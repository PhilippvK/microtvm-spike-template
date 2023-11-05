# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import fcntl
import multiprocessing
import os
import shlex
import os.path
import pathlib
import select
import shutil
import logging
import subprocess
import tarfile
import time
import distutils.util

from tvm.micro.project_api import server

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.WARNING)

SPIKE_EXE = "spike"
SPIKE_PK = "pk"

PROJECT_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"


IS_TEMPLATE = not os.path.exists(os.path.join(PROJECT_DIR, MODEL_LIBRARY_FORMAT_RELPATH))

# Used this size to pass most CRT tests in TVM.
# WORKSPACE_SIZE_BYTES = 2 * 1024 * 1024
WORKSPACE_SIZE_BYTES = 1 * 1024 * 1024

CMAKEFILE_FILENAME = "CMakeLists.txt"

# The build target given to make
BUILD_TARGET = "build/main"

ARCH = "rv32gc"
ABI = "ilp32d"
TRIPLE = "riscv32-unknown-elf"
TOOLCHAIN = "gcc"
NPROC = multiprocessing.cpu_count()
VLEN = 128
ELEN = 64


def str2bool(value, allow_none=False):
    if value is None:
        assert allow_none, "str2bool received None value while allow_none=False"
        return value
    return bool(value) if isinstance(value, (int, bool)) else bool(distutils.util.strtobool(value))


def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


class Handler(server.ProjectAPIHandler):
    BUILD_TARGET = "build/main"

    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="host",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else PROJECT_DIR / MODEL_LIBRARY_FORMAT_RELPATH,
            project_options=[
                server.ProjectOption(
                    "verbose",
                    optional=["build"],
                    type="bool",
                    default=False,
                    help="Run make with verbose output",
                ),
                server.ProjectOption(
                    "quiet",
                    optional=["build"],
                    type="bool",
                    default=True,
                    help="Supress all compilation messages",
                ),
                server.ProjectOption(
                    "debug",
                    optional=["build"],
                    type="bool",
                    default=False,
                    help="Build with debugging symbols and -O0",
                ),
                server.ProjectOption(
                    "workspace_size_bytes",
                    optional=["generate_project"],
                    type="int",
                    default=WORKSPACE_SIZE_BYTES,
                    help="Sets the value of TVM_WORKSPACE_SIZE_BYTES.",
                ),
                server.ProjectOption(
                    "arch",
                    optional=["build", "open_transport"],
                    default=ARCH,
                    type="str",
                    help="Name used ARCH.",
                ),
                server.ProjectOption(
                    "abi",
                    optional=["build"],
                    default=ABI,
                    type="str",
                    help="Name used ABI.",
                ),
                server.ProjectOption(
                    "vlen",
                    optional=["open_transport"],
                    default=VLEN,
                    type="int",
                    help="VLEN used if V-Extension is available.",
                ),
                server.ProjectOption(
                    "elen",
                    optional=["open_transport"],
                    default=ELEN,
                    type="int",
                    help="ELEN used if V-Extension is available.",
                ),
                server.ProjectOption(
                    "toolchain",
                    optional=["build"],
                    default=TOOLCHAIN,
                    choices=["gcc", "llvm"],
                    type="str",
                    help="Name used TOOLCHAIN.",
                ),
                server.ProjectOption(
                    "llvm_dir",
                    optional=["build"],
                    default=None,
                    type="str",
                    help="Path to LLVM install directory",
                ),
                server.ProjectOption(
                    "gcc_prefix",
                    optional=["build"],
                    default="",
                    type="str",
                    help="Name used COMPILER.",
                ),
                server.ProjectOption(
                    "gcc_name",
                    optional=["build"],
                    default=TRIPLE,
                    type="str",
                    help="Name used COMPILER.",
                ),
                server.ProjectOption(
                    "spike_exe",
                    required=(["open_transport"] if not SPIKE_EXE else None),
                    optional=(["open_transport"] if SPIKE_EXE else []),
                    default=SPIKE_EXE,
                    type="str",
                    help="Path to the spike (riscv-isa-sim) executable.",
                ),
                server.ProjectOption(
                    "spike_pk",
                    required=(["open_transport"] if not SPIKE_PK else None),
                    optional=(["open_transport"] if SPIKE_PK else None),
                    default=SPIKE_EXE,
                    type="str",
                    help="Path to the proxy-kernel (pk).",
                ),
                server.ProjectOption(
                    "spike_extra_args",
                    optional=["open_transport"],
                    type="str",
                    help="Additional arguments added to the spike command line.",
                ),
                server.ProjectOption(
                    "pk_extra_args",
                    optional=["open_transport"],
                    type="str",
                    help="Additional arguments added to the pk command line.",
                ),
            ],
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "CMakeLists.txt", "src")

    def _populate_cmake(
        self,
        cmakefile_template_path: pathlib.Path,
        cmakefile_path: pathlib.Path,
        memory_size: int,
        verbose: bool,
    ):
        """Generate CMakeList file from template."""

        with open(cmakefile_path, "w") as cmakefile_f:
            with open(cmakefile_template_path, "r") as cmakefile_template_f:
                for line in cmakefile_template_f:
                    cmakefile_f.write(line)
                cmakefile_f.write(
                    f"target_compile_definitions(main PUBLIC -DTVM_WORKSPACE_SIZE_BYTES={memory_size})\n"
                )
                if verbose:
                    cmakefile_f.write(f"set(CMAKE_VERBOSE_MAKEFILE TRUE)\n")

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
        # Make project directory.
        project_dir.mkdir(parents=True)
        current_dir = pathlib.Path(__file__).parent.absolute()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # Place Model Library Format tarball in the special location, which this script uses to decide
        # whether it's being invoked in a template or generated project.
        project_model_library_format_path = project_dir / MODEL_LIBRARY_FORMAT_RELPATH
        shutil.copy2(model_library_format_path, project_model_library_format_path)

        # Extract Model Library Format tarball.into <project_dir>/model.
        extract_path = project_dir / project_model_library_format_path.stem
        with tarfile.TarFile(project_model_library_format_path) as tf:
            os.makedirs(extract_path)
            tf.extractall(path=extract_path)

        # Populate CRT.
        crt_path = project_dir / "crt"
        os.mkdir(crt_path)
        for item in self.CRT_COPY_ITEMS:
            src_path = standalone_crt_dir / item
            dst_path = crt_path / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Populate CMake file
        self._populate_cmake(
            current_dir / f"{CMAKEFILE_FILENAME}.template",
            project_dir / CMAKEFILE_FILENAME,
            options.get("workspace_size_bytes", WORKSPACE_SIZE_BYTES),
            str2bool(options.get("verbose"), False),
        )
        cmake_path = project_dir / "cmake"
        os.mkdir(cmake_path)
        shutil.copytree(current_dir / "cmake", cmake_path, dirs_exist_ok=True)

        # Populate crt-config.h
        crt_config_dir = project_dir / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            current_dir / "crt_config" / "crt_config.h",
            crt_config_dir / "crt_config.h",
        )

        # Populate src/
        src_dir = project_dir / "src"
        src_dir.mkdir()
        shutil.copy2(
            current_dir / "src" / "main.cc",
            src_dir / "main.cc",
        )
        shutil.copy2(
            current_dir / "src" / "platform.cc",
            src_dir / "platform.cc",
        )

    def build(self, options):
        build_dir = PROJECT_DIR / "build"
        build_dir.mkdir()
        cmake_args = []
        debug = options.get("debug", False)
        build_type = "Debug" if debug else "Release"
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
        cmake_args.append("-DTOOLCHAIN=" + options.get("toolchain", TOOLCHAIN))
        llvm_dir = options.get("llvm_dir", None)
        if llvm_dir:
            cmake_args.append("-DLLVM_DIR=" + llvm_dir)
        cmake_args.append("-DRISCV_ARCH=" + options.get("arch", ARCH))
        cmake_args.append("-DRISCV_ABI=" + options.get("abi", ABI))
        cmake_args.append("-DRISCV_ABI=" + options.get("abi", ABI))
        cmake_args.append("-DRISCV_ELF_GCC_PREFIX=" + options.get("gcc_prefix", ""))
        cmake_args.append("-DRISCV_ELF_GCC_BASENAME=" + options.get("gcc_name", TRIPLE))
        if str2bool(options.get("quiet"), True):
            check_call(["cmake", "..", *cmake_args], cwd=build_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            check_call(["make", f"-j{NPROC}"], cwd=build_dir, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            check_call(["cmake", "..", *cmake_args], cwd=build_dir)
            check_call(["make", f"-j{NPROC}"], cwd=build_dir)

    def flash(self, options):
        pass  # Flashing does nothing on host.

    def _set_nonblock(self, fd):
        flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
        new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
        assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"

    def open_transport(self, options):
        # print("open_transport")
        isa = options.get("arch", ARCH)
        if isa is None:
            isa = ARCH
        spike_extra = options.get("spike_extra_args")
        if spike_extra in [None, ""]:
            spike_extra = []
        else:
            spike_extra = [spike_extra]
        vlen = int(options.get("vlen", VLEN))
        if vlen > 0:
            elen = int(options.get("elen", ELEN))
            assert vlen >= 128, "VLEN has to be >= 128"
            assert elen in [32, 64], "ELEN has to be either 32 or 64"
            spike_extra.append(f"--varch=vlen:{vlen},elen:{elen}")
        pk_extra = options.get("pk_extra_args")
        if pk_extra in [None, ""]:
            pk_extra = []
        else:
            pk_extra = [pk_extra]
        spike_args = [options.get("spike_exe"), f"--isa={isa}", *spike_extra, options.get("spike_pk"), *pk_extra]
        self._proc = subprocess.Popen(
            spike_args + [self.BUILD_TARGET], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self._set_nonblock(self._proc.stdin.fileno())
        self._set_nonblock(self._proc.stdout.fileno())
        return server.TransportTimeouts(
            session_start_retry_timeout_sec=0,
            session_start_timeout_sec=0,
            session_established_timeout_sec=0,
        )

    def close_transport(self):
        # print("close_transport")
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            proc.terminate()
            proc.kill()
            proc.wait()

    def _await_ready(self, rlist, wlist, timeout_sec=None, end_time=None):
        if timeout_sec is None and end_time is not None:
            timeout_sec = max(0, end_time - time.monotonic())

        rlist, wlist, xlist = select.select(rlist, wlist, rlist + wlist, timeout_sec)
        if not rlist and not wlist and not xlist:
            raise server.IoTimeoutError()

        return True

    def read_transport(self, n, timeout_sec):
        # print("read_transport", n)
        if self._proc is None:
            raise server.TransportClosedError()

        fd = self._proc.stdout.fileno()
        end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

        try:
            self._await_ready([fd], [], end_time=end_time)
            to_return = os.read(fd, n)
        except BrokenPipeError:
            to_return = 0

        if not to_return:
            self.close_transport()
            raise server.TransportClosedError()
        # print("ret", to_return)

        return to_return

    def write_transport(self, data, timeout_sec):
        # print("write_transport", data)
        if self._proc is None:
            raise server.TransportClosedError()

        fd = self._proc.stdin.fileno()
        end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

        # data_len = len(data)
        while data:
            self._await_ready([], [fd], end_time=end_time)
            try:
                num_written = os.write(fd, data)
            except BrokenPipeError:
                num_written = 0

            if not num_written:
                self.disconnect_transport()
                raise server.TransportClosedError()

            data = data[num_written:]


if __name__ == "__main__":
    server.main(Handler())
