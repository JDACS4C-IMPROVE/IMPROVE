import argparse
import configparser
import inspect
import os
from typing import Any, List, Optional, Set

from candle.helper_utils import eval_string_as_list_of_lists
from candle.parsing_utils import (
    ConfigDict,
    ParseDict,
    finalize_parameters,
    parse_common,
    parse_from_dictlist,
    registered_conf,
)

DType = Any


class Benchmark:
    """
    Class that implements an interface to handle configuration options for
    the different CANDLE benchmarks.

    It provides access to all the common configuration options and
    configuration options particular to each individual benchmark. It
    describes what minimum requirements should be specified to
    instantiate the corresponding benchmark. It interacts with the
    argparser to extract command-line options and arguments from the
    benchmark's configuration files.
    """

    def __init__(
        self,
        filepath: str,
        defmodel: str,
        framework: str,
        prog: str = None,
        desc: str = None,
        parser=None,
        additional_definitions=None,
        required=None,
    ) -> None:
        """
        Initialize Benchmark object.

        :param string filepath: ./
            os.path.dirname where the benchmark is located. Necessary to locate utils and
            establish input/ouput paths
        :param string defmodel: 'p*b*_default_model.txt'
            string corresponding to the default model of the benchmark
        :param string framework : 'keras', 'neon', 'mxnet', 'pytorch'
            framework used to run the benchmark
        :param string prog: 'p*b*_baseline_*'
            string for program name (usually associated to benchmark and framework)
        :param string desc: ' '
            string describing benchmark (usually a description of the neural network model built)
        :param argparser parser: (default None)
            if 'neon' framework a NeonArgparser is passed. Otherwise an argparser is constructed.
        """

        # Check that required system variable specifying path to data has been defined
        if os.getenv("CANDLE_DATA_DIR") is None:
            raise Exception(
                "ERROR ! Required system variable not specified.  You must define CANDLE_DATA_DIR ... Exiting"
            )

        # Check that default model configuration exits
        fname = os.path.join(filepath, defmodel)
        if not os.path.isfile(fname):
            raise Exception(
                "ERROR ! Required default configuration file not available.  File "
                + fname
                + " ... Exiting"
            )

        self.model_name = self.get_parameter_from_file(fname, "model_name")
        print("model name: ", self.model_name)

        if parser is None:
            parser = argparse.ArgumentParser(
                prog=prog,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description=desc,
                conflict_handler="resolve",
            )

        self.parser = parser
        self.file_path = filepath
        self.default_model = defmodel
        self.framework = framework

        self.registered_conf: List[ParseDict] = []
        for lst in registered_conf:
            self.registered_conf.extend(lst)

        if required is not None:
            self.required = set(required)
        else:
            self.required: Set[str] = set([])
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
        else:
            self.additional_definitions: List[ParseDict] = []

        # legacy call for compatibility with existing Benchmarks
        self.set_locals()

    def parse_parameters(self) -> None:
        """
        Functionality to parse options common for all benchmarks.

        This functionality is based on methods 'get_default_neon_parser'
        and 'get_common_parser' which are defined previously(above). If
        the order changes or they are moved, the calling has to be
        updated.
        """
        # Parse has been split between arguments that are common with the default neon parser
        # and all the other options
        self.parser = parse_common(self.parser)
        self.parser = parse_from_dictlist(self.additional_definitions, self.parser)

        # Set default configuration file
        self.conffile = os.path.join(self.file_path, self.default_model)

    def format_benchmark_config_arguments(
        self, dictfileparam: ConfigDict
    ) -> ConfigDict:
        """
        Functionality to format the particular parameters of the benchmark.

        :param ConfigDict dictfileparam: parameters read from configuration file

        :return args: parameters read from command-line
            Most of the time command-line overwrites configuration file
            except when the command-line is using default values and
            config file defines those values
        :rtype: ConfigDict
        """

        configOut = dictfileparam.copy()
        kwall = self.additional_definitions + self.registered_conf

        dtype: Optional[DType] = None
        for d in kwall:  # self.additional_definitions:
            if d["name"] in configOut.keys():
                if "type" in d:
                    dtype = d["type"]
                else:
                    dtype = None

                if "action" in d:
                    if inspect.isclass(d["action"]):
                        str_read = dictfileparam[d["name"]]
                        configOut[d["name"]] = eval_string_as_list_of_lists(
                            str_read, ":", ",", dtype
                        )
                elif d["default"] != argparse.SUPPRESS:
                    # default value on benchmark definition cannot overwrite config file
                    self.parser.add_argument(
                        "--" + d["name"],
                        type=d["type"],
                        default=configOut[d["name"]],
                        help=d["help"],
                    )

        return configOut

    def read_config_file(self, file: str) -> ConfigDict:
        """
        Functionality to read the configue file specific for each
        benchmark.

        :param string file: path to the configuration file

        :return: parameters read from configuration file
        :rtype: ConfigDict
        """

        config = configparser.ConfigParser()
        config.read(file)
        section = config.sections()
        fileParams = {}

        # parse specified arguments (minimal validation: if arguments
        # are written several times in the file, just the first time
        # will be used)
        for sec in section:
            for k, v in config.items(sec):
                # if not k in fileParams:
                if k not in fileParams:
                    fileParams[k] = eval(v)

        fileParams = self.format_benchmark_config_arguments(fileParams)

        # print(fileParams)

        return fileParams

    def get_parameter_from_file(self, absfname, param):
        """
        Functionality to extract the value of one parameter from the configuration file given. Execution is terminated if the parameter specified is not found in the configuration file.

        :param string absfname: filename of the the configuration file including absolute path.

        :param string param: parameter to extract from configuration file.

        :return: a string with the value of the parameter read from the configuration file.
        :rtype: string
        """

        aux = ""
        with open(absfname, "r") as fp:
            for line in fp:
                # search string
                if param in line:
                    aux = line.split("=")[-1].strip("'\n ")
                    # don't look for next lines
                    break
        if aux == "":
            raise Exception(
                "ERROR ! Parameter "
                + param
                + " was not found in file "
                + absfname
                + "... Exiting"
            )

        return aux

    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark.

        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing \
            the additional parameters for the benchmark.
        """
        pass

    def check_required_exists(self, gparam: ConfigDict) -> None:
        """
        Functionality to verify that the required model parameters have been
        specified.
        """

        key_set = set(gparam.keys())
        intersect_set = key_set.intersection(self.required)
        diff_set = self.required.difference(intersect_set)

        if len(diff_set) > 0:
            raise Exception(
                "ERROR ! Required parameters are not specified.  These required parameters have not been initialized: "
                + str(sorted(diff_set))
                + "... Exiting"
            )


def create_params(
    file_path=None,
    default_model=None,
    framework=None,
    prog_name=None,
    desc=None,
    additional_definitions=None,
    required=None,
):

    print("Generating parameters for standard benchmark\n")

    # file_path = os.path.dirname(os.path.realpath(__file__))
    tmp_bmk = Benchmark(
        file_path,
        default_model,
        framework,
        prog_name,
        desc,
        additional_definitions=additional_definitions,
        required=required,
    )

    params = finalize_parameters(tmp_bmk)

    return params
