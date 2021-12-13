import Augmentor

def main():
    num_of_samples=888
    
    p = Augmentor.Pipeline("dataset/weld")
    p.rotate(probability=0.8, max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=0.8)
    p.sample(num_of_samples)


if __name__ == '__main__':
    main()
    print('Done augment.')