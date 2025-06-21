input_a="""
 0.926, -0.455, -0.233, -0.045,  0.582,  0.625,  0.058, -0.040, 
"""

input_b="""
-0.166,  0.996,  0.441,  0.863, -1.000, -0.742, -0.395,  1.000, 
"""

input_a_parsed = [x.strip() for x in input_a.split(",") if x.strip()]
input_b_parsed = [x.strip() for x in input_b.split(",") if x.strip()]

dot_product = sum(float(a) * float(b) for a, b in zip(input_a_parsed, input_b_parsed))

print(f'Dot product is: {dot_product:.3f}')