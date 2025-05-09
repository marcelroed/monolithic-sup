trait Node:
    fn fwd[OutputType: AnyType](self) -> OutputType:
        pass

struct AddNode[Left: Node, Right: Node](Node):
    fn fwd[AddType: Add](self, left: Left, right: Right) -> Node:
        return left.fwd() + right.fwd()
    pass

struct ConstNode[Value: Int](Node):
    pass


fn main():
    alias graph = AddNode[
        ConstNode[1],
        ConstNode[2],
    ]

    
