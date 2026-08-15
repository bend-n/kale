# `kale`

kale is a stack based left-to-right interpreted array programming language

this means every value is immediately pushed to the stack
and every function will pop items off of the stack
there are a few functions for stack manipulation.

kale operates with a inferred "type"ing system, as in each function declares it's "signature" (items popped off stack -> items pushed to stack) and it goes from there

most arithmetic / basic operations is pervasive, meaning that when running an operation over an array, it will compare them by each individual value.

so when comparing arrays to arrays you get an array of booleans; same for arrays and values. comparing a value to a value you will get a single value.

comparisons are reversed, for intuition sake.
