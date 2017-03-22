def convert_variables_to_constants(sess, input_graph_def, output_node_names,
                                   variable_names_whitelist=None):
  """Replaces all the variables in a graph with constants of the same values.
  If you have a trained graph containing Variable ops, it can be convenient to
  convert them all to Const ops holding the same values. This makes it possible
  to describe the network fully with a single GraphDef file, and allows the
  removal of a lot of ops related to loading and saving the variables.
  Args:
    sess: Active TensorFlow session containing the variables.
    input_graph_def: GraphDef object holding the network.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_whitelist: The set of variable names to convert (by default,
                              all variables are converted).
  Returns:
    GraphDef containing a simplified version of the original.
  """
  found_variables = {}
  variable_names = []
  variable_dict_names = []
  for node in input_graph_def.node:
    if node.op == "Assign":
      variable_name = node.input[0]
      if (variable_names_whitelist is not None and
          variable_name not in variable_names_whitelist):
        continue
      variable_dict_names.append(variable_name)
      variable_names.append(variable_name + ":0")
  if variable_names:
    returned_variables = sess.run(variable_names)
  else:
    returned_variables = []
  found_variables = dict(zip(variable_dict_names, returned_variables))
  logging.info("Frozen %d variables." % len(returned_variables))

  # This graph only includes the nodes needed to evaluate the output nodes, and
  # removes unneeded nodes like those involved in saving and assignment.
  inference_graph = extract_sub_graph(input_graph_def, output_node_names)

  output_graph_def = graph_pb2.GraphDef()
  how_many_converted = 0
  for input_node in inference_graph.node:
    output_node = graph_pb2.NodeDef()
    if input_node.name in found_variables:
      output_node.op = "Const"
      output_node.name = input_node.name
      dtype = input_node.attr["dtype"]
      data = found_variables[input_node.name]
      output_node.attr["dtype"].CopyFrom(dtype)
      output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
          tensor=tensor_util.make_tensor_proto(data,
                                               dtype=dtype.type,
                                               shape=data.shape)))
      how_many_converted += 1
    else:
      output_node.CopyFrom(input_node)
    output_graph_def.node.extend([output_node])
  print("Converted %d variables to const ops." % how_many_converted)
return output_graph_def
