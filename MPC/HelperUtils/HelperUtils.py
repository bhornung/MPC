#-----------------------
def _assert_type(obj, allowed_types, obj_name):

  try:
    allowed_type_string = " ".join(map(lambda x: str(x), allowed_types))
    allowed_type_tuple = tuple(allowed_types)
  else:
    allowed_type_string = str(allowed_types)
    allowed_type_tuple = tuple([allowed_types])

  if not isinstance(obj, allowed_type_tuple):
    raise ValueError("Parameter {0} should be of type {1}.".format(obj_name, allowed_type_string))


