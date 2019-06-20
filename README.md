# Bidirectional-LSTM-POS-Tagger
# Introduction

The goal of this coding assignment to get you expertise in TensorFlow, especially in developing code from the grounds-up. This assignment will be not much hand-held: you have to
do most things from scratch, including creating a tf.Session. In this task, you will be implementing a sequence-to-sequence Recursive Neural Network (RNN) model in TensorFlow.
You will be using the same data from the HMM Coding Assignment 3 for Part-of-Speech
tagging. In particular, you are expected to:
• Populate the starter code in designated places marked by TODO(student).
– The starter code is on: http://sami.haija.org/cs544/DL5/starter.py.
• Write code for reading the data files and producing numpy arrays that will be used for
training [this has to produce expected outcomes, as measured by grading scripts]. This
should be implemented in class DatasetReader.
• Write code for constructing and training the model [here, you should be creative, per
grading scheme below!]. This should be implemented in class SequenceModel.
– You can optionally fill-in the main() code-block, so that you can run locally (or in
vocareum without submitting i.e. for debugging). However, the main() function
will not be run through the submission script.
– You must implement the functions that are annotated in the starter code. The
grading script will train your model for exactly K seconds1
. Therefore, you must
explore good hyperparameters for this training budget [e.g. batch size, learning
rate], which are always a function of your model architecture and the training
algorithm (there is no one answer that fits all!)
