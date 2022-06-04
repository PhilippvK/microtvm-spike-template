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
.. _tutorial-micro-autotune:

Autotuning with micro TVM
=========================
**Authors**:
`Andrew Reusch <https://github.com/areusch>`_,
`Mehrdad Hessar <https://github.com/mehrdadh>`_

This tutorial explains how to autotune a model using the C runtime.
"""

import os
import numpy as np
import subprocess
from pathlib import Path

import tvm

import logging
import sys
logging.basicConfig(level="WARNING", stream=sys.stdout)

DIR = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
ETISS_DIR = os.environ.get("ETISS_DIR", None)
assert ETISS_DIR, "Missing environment variable: ETISS_DIR"
RISCV_DIR = os.environ.get("RISCV_DIR", None)
assert RISCV_DIR, "Missing environment variable: RISCV_DIR"
# ETISSVP_SCRIPT = os.environ.get("ETISSVP_SCRIPT", str(DIR / "template_project" / "scripts" / "run.sh"))
ETISSVP_SCRIPT = os.environ.get("ETISSVP_SCRIPT", str(Path(ETISS_DIR) / "bin" / "run_helper.sh"))
assert ETISSVP_SCRIPT, "Missing environment variable: ETISSVP_SCRIPT"
ETISSVP_INI = os.environ.get("ETISSVP_INI", str(DIR / "template_project" / "scripts" / "memsegs.ini"))
assert ETISSVP_INI, "Missing environment variable: ETISSVP_INI"


project_options = {
    "project_type": "host_driven",
    # "verbose": False,
    "verbose": True,
    # "debug": False,
    "debug": True,
    "transport": True,
    "etiss_path": ETISS_DIR,
    "riscv_path": RISCV_DIR,
    "etissvp_script": ETISSVP_SCRIPT,
    "etissvp_script_args": "plic clint uart v" + (" -i" + ETISSVP_INI if ETISSVP_INI else "")
}

####################
# Defining the model
####################

data_shape = (1, 3, 10, 10)
weight_shape = (6, 3, 5, 5)

data = tvm.relay.var("data", tvm.relay.TensorType(data_shape, "float32"))
weight = tvm.relay.var("weight", tvm.relay.TensorType(weight_shape, "float32"))

y = tvm.relay.nn.conv2d(
    data,
    weight,
    padding=(2, 2),
    kernel_size=(5, 5),
    kernel_layout="OIHW",
    out_dtype="float32",
)
f = tvm.relay.Function([data, weight], y)

relay_mod = tvm.IRModule.from_expr(f)
relay_mod = tvm.relay.transform.InferType()(relay_mod)

weight_sample = np.random.rand(
    weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
).astype("float32")
params = {"weight": weight_sample}

#######################
# Defining the target #
#######################

# Compiling for virtual hardware
# --------------------------------------------------------------------------
TARGET = tvm.target.target.micro("host")
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
#TARGET = tvm.target.target.riscv_cpu("bare_etiss_processor")
BOARD = "bare_etiss_processor"

#########################
# Extracting tuning tasks
#########################

pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, TARGET)
assert len(tasks) > 0

######################
# Configuring microTVM
######################

# Compiling for virtual hardware
# --------------------------------------------------------------------------
module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=Path(str(DIR / "template_project")),
    project_options=project_options,
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=False,
    build_func=tvm.micro.autotvm_build_func,
    runtime=RUNTIME,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

################
# Run Autotuning
################

num_trials = 10
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file("microtvm_autotune.log.txt"),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
        ],
        si_prefix="M",
    )

############################
# Timing the untuned program
############################
with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()

# Compiling for virtual hardware
# --------------------------------------------------------------------------
project = tvm.micro.generate_project(
    str(DIR / "template_project"),
    lowered,
    temp_dir / "project",
    project_options,
)

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    #debug_module = tvm.micro.create_local_debug_executor(
    #    lowered.get_graph_json(), session.get_system_lib(), session.device
    #)
    debug_module = tvm.micro.create_local_graph_executor(
        lowered.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered.get_params())
    print("########## Build without Autotuning ##########")
    debug_module.run()
    del debug_module

##########################
# Timing the tuned program
##########################

with tvm.autotvm.apply_history_best("microtvm_autotune.log.txt"):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()

# Compiling for virtual hardware
# --------------------------------------------------------------------------
project = tvm.micro.generate_project(
    str(DIR / "template_project"),
    lowered_tuned,
    temp_dir / "project",
    project_options,
)

project.build()
project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered_tuned.get_params())
    print("########## Build with Autotuning ##########")
    #debug_module.run()
    del debug_module