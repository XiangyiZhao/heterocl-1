import heterocl as hcl
import numpy as np

def test_reuse_blur_x_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 10), lambda y, x: hcl.select((x<8), (A[y, x] + A[y, x+1] + A[y, x+2]), A[y, x]))
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 10), dtype="int")
    np_C = np.zeros((10, 10), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
          if (x<8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]
          else:
            np_C[y][x] = np_A[y][x]
        

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)
    
def test_reuse_blur_x_tensor_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10))
    X = hcl.compute((10, 10), lambda y, x: A[y, x])
    B = hcl.compute((10, 10), lambda y, x: hcl.select((x<8), (X[y, x] + X[y, x+1] + X[y, x+2]), X[y, x]))
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(X, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 10), dtype="int")
    np_C = np.zeros((10, 10), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
          if (x<8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]
          else:
            np_C[y][x] = np_A[y][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_y_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 10), lambda y, x: hcl.select((y<8), (A[y, x] + A[y+1, x] + A[y+2, x]), A[y, x]))
    s = hcl.create_schedule([A, B])
    print(hcl.lower(s))
    print("--------------")
    RB = s.reuse_at(A, s[B], B.axis[0])
    print(hcl.lower(s))
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 10), dtype="int")
    np_C = np.zeros((10, 10), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
          if (y<8):
            np_C[y][x] = np_A[y][x] + np_A[y+1][x] + np_A[y+2][x]
          else:
            np_C[y][x] = np_A[y][x]
            
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_y_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute((10, 10), lambda y, x: hcl.select(hcl.and_(x<8, y<8), (A[y, x] + A[y+1, x+1] + A[y+2, x+2]), A[y, x]))
    s = hcl.create_schedule([A, B])
    RB_y = s.reuse_at(A, s[B], B.axis[0], "RB_y")
    RB_x = s.reuse_at(RB_y, s[B], B.axis[1], "RB_x")
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 10), dtype="int")
    np_C = np.zeros((10, 10), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
          if (x<8 and y<8):
            np_C[y][x] = np_A[y][x] + np_A[y+1][x+1] + np_A[y+2][x+2]
          else:
            np_C[y][x] = np_A[y][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_3D_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10, 2))
    B = hcl.compute((10, 10, 2), lambda y, x, c: hcl.select((x<8), (A[y, x, c] + A[y, x+1, c] + A[y, x+2, c]),  A[y, x, c]))
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10, 2))
    np_B = np.zeros((10, 10, 2), dtype="int")
    np_C = np.zeros((10, 10, 2), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
            for c in range(0, 2):
              if (x<8):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y][x+1][c] + np_A[y][x+2][c]
              else:
                np_C[y][x][c] = np_A[y][x][c]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_y_3D_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10, 2))
    B = hcl.compute((10, 10, 2), lambda y, x, c: hcl.select((y<8), (A[y, x, c] + A[y+1, x, c] + A[y+2, x, c]), A[y, x, c]))
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10, 2))
    np_B = np.zeros((10, 10, 2), dtype="int")
    np_C = np.zeros((10, 10, 2), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
            for c in range(0, 2):
              if (y<8):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y+1][x][c] + np_A[y+2][x][c]
              else:
                np_C[y][x][c] = np_A[y][x][c]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_y_3D_hcl_select():
    hcl.init()
    A = hcl.placeholder((10, 10, 2), "A")
    B = hcl.compute((10, 10, 2), lambda y, x, c: hcl.select(hcl.and_(x<8, y<8), (A[y, x, c] + A[y+1, x+1, c] + A[y+2, x+2, c]),  A[y, x, c]))
    s = hcl.create_schedule([A, B])
    RB_y = s.reuse_at(A, s[B], B.axis[0], "RB_y")
    RB_x = s.reuse_at(RB_y, s[B], B.axis[1], "RB_x")
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10, 2))
    np_B = np.zeros((10, 10, 2), dtype="int")
    np_C = np.zeros((10, 10, 2), dtype="int")

    for y in range(0, 10):
        for x in range(0, 10):
            for c in range(0, 2):
              if (x<8) and (y<8):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y+1][x+1][c] + np_A[y+2][x+2][c]
              else:
                np_C[y][x][c] = np_A[y][x][c]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)


#test_reuse_blur_x_hcl_select()
#test_reuse_blur_x_tensor_hcl_select()
test_reuse_blur_y_hcl_select()
#test_reuse_blur_x_y_hcl_select()
#test_reuse_blur_x_3D_hcl_select()
#test_reuse_blur_y_3D_hcl_select()
#test_reuse_blur_x_y_3D_hcl_select()

#Padding:
#new_img = [][]
#n = expr_list.size()
#for i in range(height+n):
#    for j in range(width+n):
#        if(i >= height and j >= width ):
#            new_img[i][j] = 0
#        else:
#            old_img[i][j] = img[i][j]
