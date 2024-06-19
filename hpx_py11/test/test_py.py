import sys
import gaussian_process as a

import time

if __name__ == "__main__":

        print(sys.argv)

        n_train = 300
        n_test = 700
        n_tile_size = 100

        try:
                n_tiles = a.compute_train_tiles(n_train, n_tile_size)
                print("Train Tiles:", n_tiles)
        except RuntimeError as e:
                print(e)

        print("Number of training tiles: ", n_tiles)
        m_tiles, m_tile_size = a.compute_test_tiles(n_test, n_tiles, n_tile_size)
        print(f"m_tiles: {m_tiles}, m_tile_size: {m_tile_size}")

        print(f"# Train samples: {n_train}")
        print(f"# Test samples: {n_test}")
        print(f"Tile size in N dimension: {n_tile_size}")
        print(f"Tile size in M dimension: {m_tile_size}")
        print(f"Tiles in N dimension: {n_tiles}")
        print(f"Tiles in M dimension: {m_tiles}")

        hpar = a.Hyperparameters(learning_rate=0.1, opt_iter=2)
        print(hpar.beta1)
        print(hpar)

        f1_path = ("/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt")
        train_in = a.GP_data(f1_path, n_train)
        print("traning input")
        a.print(train_in.data, 0, 5, ", ")
        f2_path = ("/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt")
        train_out = a.GP_data(f2_path, n_train)
        f3_path = ("/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt")
        test_in = a.GP_data(f3_path, n_test)
        try:
                data = train_in.data
                # print("Data:", data)

                # Call print_slice function
                # Example 1: Print the entire vector
                a.print(data, 0, 3)
                print("data path:", train_in.file_path)
                print("number samples:", train_in.n_samples)

                # Example 2: Print elements from index 1 to 3 (excluding 3)
                # a.print_slice(data, slice(1, 3), ", ")
        except AttributeError as e:
                print(f"Error accessing train_data.data: {e}")

        ###### GP object ######
        gp = a.GP(train_in.data, train_out.data, n_tiles, n_tile_size, trainable=[False, False, True])
        print("traning input")
        a.print(gp.get_input_data(), 0, 5, ", ")
        print(gp)
        start = time.time()
        a.start_hpx(sys.argv, 2)
        end = time.time()
        print("hpx start time: ", end - start)
        losses = gp.optimize(hpar)
        print("loss")
        a.print(losses)
        pr = gp.predict(test_in.data, m_tiles, m_tile_size)
        print("predict: ")
        # a.print(pr[0])
        print("uncertainty: ")
        # a.print(pr[1])
        print(gp)
        s = time.time()
        a.stop_hpx()
        e = time.time()
        print("hpx shutdown time: ", e - s)
