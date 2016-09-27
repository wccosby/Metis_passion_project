# Passion Project: Chat.ly
Final Project for Metis Data Science Bootcamp, automated customer service agent.

This repository is meant to be a testing ground for variations on dynamic memory networks.  For a production version of this project built out in Python Flask and supporting AWS see chatly_web repository.

# Project Objectives
* Understand state of the art for dynamic memory networks.
* Implement a base Dynamic memory network to be used and expanded on for my final project.

# Dynamic Memory Network
* Paper
    * http://arxiv.org/abs/1506.07285
* Used this code and blog post as an extremely helpful guideline
    * https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/
    * https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
* Data
    * https://research.facebook.com/research/babi/
* I used Tensorflow to implement the network

# File Structure
* main.py contains code to run network and pass in various options and hyper parameters
* read_data.py and utils.py are from yerevann DMN and take care of formatting the data for the training and testing.
* models/n2n_DMN/dmn.py contains the primary code for the network itself.
* log/ contains information used by tensorboard to graph the network
* save/ contains previously trained models
