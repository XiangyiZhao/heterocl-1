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
    RB = s.reuse_at(A, s[B], B.axis[0])
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


def test_reuse_blur_x_hcl_irregular():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 8), lambda y, x: A[y, 2*x] + A[y, 2*x+1] + A[y, 2*x+2])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][2*x] + np_A[y][2*x+1] + np_A[y][2*x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)
    
def test_reuse_blur_x_hcl_irregular_x():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 8), lambda y, x: A[y, x*x] + A[y, x*x+1] + A[y, x*x+2])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x*x] + np_A[y][x*x+1] + np_A[y][x*x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

#This in the logic is said to be a "Irregular error" but comes out as a "region not determined" error
#Since min_diff is different everytime: x
def test_reuse_blur_x_hcl_irregular_x_c():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 8), lambda y, x: A[y, x*x] + A[y, x*(x+1)] + A[y, x*(x+2)])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x*x] + np_A[y][x*(x+1)] + np_A[y][x*(x+2)]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)
       

def test_reuse_blur_y_hcl_irregular():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((8, 10), lambda y, x: A[2*y, x] + A[2*y+1, x] + A[2*y+2, x])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 10), dtype="int")
    np_C = np.zeros((8, 10), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            np_C[y][x] = np_A[2*y][x] + np_A[2*y+1][x] + np_A[2*y+2][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_y_hcl_irregular_y():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((8, 10), lambda y, x: A[y*y, x] + A[y*y+1, x] + A[y*y+2, x])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 10), dtype="int")
    np_C = np.zeros((8, 10), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            np_C[y][x] = np_A[y*y][x] + np_A[y*y+1][x] + np_A[y*y+2][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)
    
def test_reuse_blur_y_hcl_irregular_y_c():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((8, 10), lambda y, x: A[y*y, x] + A[y*(y+1), x] + A[y*(y+2), x])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 10), dtype="int")
    np_C = np.zeros((8, 10), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            np_C[y][x] = np_A[y*y][x] + np_A[y*(y+1)][x] + np_A[y*(y+2)][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)    

#This in the logic is said to be a "No reuse is found" 
#but from the logic we see that if max_expr <= next_min, then no reuse is found
#In this case max_expr = x+2, next_min = ((x+1)+1)
def test_reuse_blur_x_hcl_no_reuse():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 8), lambda y, x: A[y, x+1] + A[y, x+2])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x+1] + np_A[y][x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

#Without reuse will work but with reuse will not
def test_reuse_blur_y_hcl_no_reuse():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((7, 10), lambda y, x: A[y+1, x] + A[y+3, x])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((7, 10), dtype="int")
    np_C = np.zeros((7, 10), dtype="int")

    for y in range(0, 7):
        for x in range(0, 10):
            np_C[y][x] = np_A[y+1][x] + np_A[y+3][x]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()
    print(np_B)
    print('-')
    print(np_C)
  #  assert np.array_equal(np_B, np_C)

#How to reach "no reuse dimension found in the body" error 
def test_reuse_blur_x_hcl_select_no_dimension():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 8), lambda y, x: A[y, x-1] + A[y, x] + A[y, x])
    s = hcl.create_schedule([A, B])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")
    f(hcl_A, hcl_B)
    
#    for y in range(0, 10):
#        for x in range(0, 8):
#          if (x<8):
#            np_C[y][x] = np_A[y][x] + np_A[y][x] + np_A[y][x]
#          else:
#            np_C[y][x] = np_A[y][x]
#        
#
#    hcl_A = hcl.asarray(np_A)
#    hcl_B = hcl.asarray(np_B)
#
#    f(hcl_A, hcl_B)
#
#    np_B = hcl_B.asnumpy()
#
#    assert np.array_equal(np_B, np_C)  

#test_reuse_blur_x_hcl_select()
#test_reuse_blur_x_tensor_hcl_select()
#test_reuse_blur_y_hcl_select()
#test_reuse_blur_x_y_hcl_select()
#test_reuse_blur_x_3D_hcl_select()
#test_reuse_blur_y_3D_hcl_select()
#test_reuse_blur_x_y_3D_hcl_select()

#test_reuse_blur_x_hcl_irregular()
#test_reuse_blur_x_hcl_irregular_x()
#test_reuse_blur_x_hcl_irregular_x_c()
#test_reuse_blur_y_hcl_irregular()
#test_reuse_blur_y_hcl_irregular_y()
#test_reuse_blur_y_hcl_irregular_y_c()
#test_reuse_blur_x_hcl_no_reuse()
test_reuse_blur_y_hcl_no_reuse()
#test_reuse_blur_x_hcl_select_no_dimension()

#Padding:
#new_img = [][]
#n = expr_list.size()
#for i in range(height+n):
#    for j in range(width+n):
#        if(i >= height and j >= width ):
#            new_img[i][j] = 0
#        else:
#            old_img[i][j] = img[i][j]
