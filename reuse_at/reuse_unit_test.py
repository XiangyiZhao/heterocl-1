import heterocl as hcl
import numpy as np

def test_reuse_blur_x():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 7), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2]+ A[y, x+3])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)
    print(hcl.lower(s))
    
    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 7), dtype="int")
    np_C = np.zeros((10, 7), dtype="int")

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    for y in range(0, 10):
        for x in range(0, 7):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2] + np_A[y][x+3]

    np_B = hcl_B.asnumpy()
    assert np.array_equal(np_B, np_C)


def test_reuse_blur_x_tensor():
    hcl.init()
    A = hcl.placeholder((10, 10))
    X = hcl.compute((10, 10), lambda y, x: A[y, x])
    B = hcl.compute((10, 8), lambda y, x: X[y, x] + X[y, x+1] + X[y, x+2])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(X, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_y():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((8, 10), lambda y, x: A[y, x] + A[y+1, x] + A[y+2, x])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 10), dtype="int")
    np_C = np.zeros((8, 10), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            np_C[y][x] = np_A[y][x] + np_A[y+1][x] + np_A[y+2][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_y():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute((8, 8), lambda y, x: A[y, x] + A[y+1, x+1] + A[y+2, x+2], "B")
    s = hcl.create_schedule([A, B])
    RB_y = s.reuse_at(A, s[B], B.axis[0], "RB_y")
    RB_x = s.reuse_at(RB_y, s[B], B.axis[1], "RB_x")
    f = hcl.build(s)
    print(hcl.lower(s))
    
    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 8), dtype="int")
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y+1][x+1] + np_A[y+2][x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)
    
#test_reuse_blur_x()
test_reuse_blur_x_y()