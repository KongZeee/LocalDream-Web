
try:
    import torch_directml
    print("torch_directml imported OK")
    print("has memory_allocated:", hasattr(torch_directml, 'memory_allocated'))
    print("has get_device_properties:", hasattr(torch_directml, 'get_device_properties'))
    # Try calling it
    try:
        used = torch_directml.memory_allocated()
        print(f"memory_allocated() = {used}")
    except Exception as e:
        print(f"memory_allocated() error: {e}")
    try:
        props = torch_directml.get_device_properties(0)
        print(f"get_device_properties(0) = {props}")
    except Exception as e:
        print(f"get_device_properties error: {e}")
except Exception as e:
    print(f"import error: {e}")
