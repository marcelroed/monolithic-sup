from collections.string import String

@value
struct Primitive(Stringable):
    var i: Int32
    var j: Int32

    @no_inline
    fn __str__(self) -> String:
        var i = self.i
        var j = self.j
        var s1 = String("Primitive(i: ", i, ", j: ", j, ")")
        return s1



fn main() raises:
    x = Primitive(1, 2)
    print(String(x))
    
