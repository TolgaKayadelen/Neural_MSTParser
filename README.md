#### Codebase for my PhD dissertation [Joint Learning of Syntax and Argument Structure in Dependency Parsing](https://open.metu.edu.tr/handle/11511/102788)

- Implementation is done via Tensorflow 2.0, and using the [Bazel](https://bazel.build/) build system. 

###### Contents
 - The `/parser` directory contains implementation of the dependency parsing models described Ch. 5 of the thesis. All parsers subclass from `base_parser.py`.
 - The `/ranker` directory contains implementation of Reinforcement Learning Reranker model described in Ch. 7 of the thesis
 - The `/ìnput` directory contains preprocessing libraries. We convert data to tf.data.Dataset examples. 
 - The `/proto` directory contains protobuf specification that we employ for representating dependency data.
 - The `/transformer` directory contains BERT models finetuned on seq2seq dependency label prediction task and the finetuning code.


More instructions on how to build, run, and train the various parsers implemented here will follow. 