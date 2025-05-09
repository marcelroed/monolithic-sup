trait Tensor:
    """
    Tensors can be tracers or values.
    """
    pass

trait TensorImplementation:
    """
    """
    # fn apply[*ArgTypes: AnyType](self, *args: *ArgTypes) -> AnyType:
    #     pass
    # fn name(self) -> String:
    #     pass
    pass


struct Add(TensorImplementation):
    pass


fn find_top_trace()


fn bind[Prim: TensorImplementation](
    *args: ShapedArray,
) -> ShapedArray:
    top_trace = find_top_trace(*args)


struct ShapedArray:
    """
    Array for evaluation.
    """
    pass



fn neg(x: Tensor) -> Tensor:
    return bind[Neg](x)

fn add(x: Tensor, y: Tensor) -> Tensor:
    return bind[Add](x, y)