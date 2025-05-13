for shape in range(1, 6):
    for height in range(1, 4):
        for vis_col in range(1, 6):
            label = ((shape - 1) * 15 + (height - 1) * 5 + (vis_col - 1))
            print(label, '-', shape, height, vis_col)