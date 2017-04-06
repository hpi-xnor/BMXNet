/*!
 * Copyright (c) 2017 by Contributors
 * \file main.cpp
 * \brief model-converter main
 * \author HPI-DeepLearning
*/
#include <stdio.h>
#include <libgen.h>
#include <fstream>

#include <mxnet/ndarray.h>

#include "../src/xnor_cpu.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using mxnet::op::xnor_cpu::BITS_PER_BINARY_WORD;
using mxnet::op::xnor_cpu::BINARY_WORD;


void convert_to_binary(mxnet::NDArray& array) {
  assert(mshadow::mshadow_sizeof(array.dtype()) == sizeof(BINARY_WORD));
  assert(array.shape().ndim() == 4); // adjust for FC, flatten
  assert(array.shape()[1] % BITS_PER_BINARY_WORD == 0);
  nnvm::TShape binarized_shape(1);
  int size = array.shape().Size();
  binarized_shape[0] = size / BITS_PER_BINARY_WORD;
  mxnet::NDArray temp(binarized_shape, mxnet::Context::CPU(), false, array.dtype());
  mxnet::op::xnor_cpu::get_binary_row((float*) array.data().dptr_, (BINARY_WORD*) temp.data().dptr_, size);
  array = temp;
}

int convert_params_file(const std::string& input_file, const std::string& output_file) {
  std::vector<mxnet::NDArray> data;
  std::vector<std::string> keys;

  std::cout << "loading " << input_file << "..." << std::endl;
  { // loading params file into data and keys
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(input_file.c_str(), "r"));
    mxnet::NDArray::Load(fi.get(), &data, &keys);
  }

  const std::string filter_string("qconvolution");
  auto containsSubString = [filter_string](std::string s) {
    return s.find(filter_string) != std::string::npos;}; //@todo: add FC

  auto iter = std::find_if(keys.begin(),
                           keys.end(),
                           containsSubString);

  //Use a while loop, checking whether iter is at the end of myVector
  //Do a find_if starting at the item after iter, std::next(iter)
  while (iter != keys.end())
  {
    std::cout << "converting weights " << *iter << "..." << std::endl;
    convert_to_binary(data[iter - keys.begin()]);
    iter = std::find_if(std::next(iter),
                        keys.end(),
                        containsSubString);
  }


  { // saving params back to *_converted
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(output_file.c_str(), "w"));
    mxnet::NDArray::Save(fo.get(), data, keys);
  }
  std::cout << "wrote converted params to " << output_file << std::endl;
  return 0;
}

int convert_json_file(const std::string& input_fname, const std::string& output_fname) {
  std::cout << "loading " << input_fname << "..." << std::endl;
  std::string json;
  {
    std::ifstream stream(input_fname);
    if (!stream.is_open()) {
      std::cout << "cant find json file at " + input_fname << std::endl;
      return -1;
    }
    std::stringstream buffer;
    buffer << stream.rdbuf();
    json = buffer.str();
  }

  rapidjson::Document d;
  d.Parse(json.c_str());

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);

  {
    std::ofstream stream(output_fname);
    if (!stream.is_open()) {
      std::cout << "cant find json file at " + output_fname << std::endl;
      return -1;
    }
    std::string output = buffer.GetString();
    stream << output;
    stream.close();
  }

  std::cout << "wrote converted json to " << output_fname << std::endl;

  return 0;
}

int main(int argc, char ** argv){
  if (argc != 2) {
    std::cout << "usage: " + std::string(argv[0]) + " <mxnet *.params file>" << std::endl;
    return -1;
  }

  const std::string params_file(argv[1]);

  const std::string path(dirname(argv[1]));
  const std::string params_file_name(basename(argv[1]));
  std::string base_name = params_file_name;
  base_name.erase(base_name.rfind('-')); // watchout if no '-'
  const std::string output_name(path + "/" + "binarized_" + params_file_name);

  if (convert_params_file(params_file, output_name) != 0) {
    return -1;
  }

  const std::string json_file_name(path + "/"                + base_name + "-symbol.json");
  const std::string json_out_fname(path + "/" + "binarized_" + base_name + "-symbol.json");

  if (convert_json_file(json_file_name, json_out_fname) != 0) {
    return -1;
  }

  return 0;
}