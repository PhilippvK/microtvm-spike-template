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
"""
.. _tutorial_micro_tflite:

2. microTVM TFLite Tutorial
===========================
**Author**: `Tom Gall <https://github.com/tom-gall>`_

This tutorial is an introduction to working with microTVM and a TFLite
model with Relay.
"""

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_dependencies.rst
#


import os

# By default, this tutorial runs on x86 CPU using TVM's C runtime. If you would like
# to run on real Zephyr hardware, you must export the `TVM_MICRO_USE_HW` environment
# variable. Otherwise (if you are using the C runtime), you can skip installing
# Zephyr. It takes ~20 minutes to install Zephyr.
use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_zephyr.rst
#

######################################################################
# Import Python dependencies
# -------------------------------
#
import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
import tvm.contrib.utils
from tvm.micro import export_model_library_format
from tvm.contrib.download import download_testdata

template_project_path = pathlib.Path("/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/src/microtvm_spike/template_project/")
project_options = {
    "toolchain": "llvm",
    "llvm_dir": "/var/tmp/ga87puy/ll/llvm-project/install/",
    "gcc_prefix": "/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/install/riscv_gcc_vext/",
    "spike_exe": "/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/install/spike/spike",
    "spike_pk": "/var/tmp/ga87puy/mlonmcu/mlonmcu/workspace/deps/install/spikepk/pk",
    "arch": "rv32gcv",
    # "arch": "rv32gc",
    "vlen": 128,
    "elen": 64,
}


# Create a temporary directory
# temp_dir = tvm.contrib.utils.tempdir()
temp_dir = pathlib.Path("/tmp/tmptjgioiqp")
generated_project_dir = temp_dir / "generated-project"
# generated_project = tvm.micro.generate_project(
#     template_project_path, module, generated_project_dir, project_options
# )
generated_project = tvm.micro.GeneratedProject.from_directory(generated_project_dir, project_options)

# Build and flash the project
# generated_project.build()
# generated_project.flash()

graph = generated_project_dir / "model/executor-config/graph/default.graph"
params = generated_project_dir / "model/parameters/default.params"

with open(params, "rb") as params_file:
    params = relay.load_param_dict(params_file.read())
with open(graph) as graph_file:
    graph = graph_file.read()


######################################################################
# Next, establish a session with the simulated device and run the
# computation. The `with session` line would typically flash an attached
# microcontroller, but in this tutorial, it simply launches a subprocess
# to stand in for an attached microcontroller.

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    # graph_mod = tvm.micro.create_local_graph_executor(
    #     module.get_graph_json(), session.get_system_lib(), session.device
    # )
    graph_mod = tvm.micro.create_local_debug_executor(
        graph, session.get_system_lib(), session.device
    )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**params)

    # The model consumes a single float32 value and returns a predicted sine value.  To pass the
    # input value we construct a tvm.nd.array object with a single contrived number as input. For
    # this model values of 0 to 2Pi are acceptable.
    # graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).numpy()
    print("result is: " + str(tvm_output))
