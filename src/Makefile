CC = mpicc
CFLAGS = -O3 -std=c99 -g -Wall

sec: main.c imagen.c ga.c derivados_mpi.c
	$(CC) $(CFLAGS) main.c imagen.c ga.c derivados_mpi.c -o sec -DTIME -DDEBUG -lm
test_sec_small_256:
	mpirun -np 8 -host marte,mercurio -npernode 4 ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_small_256.ppm 7000 256
test_sec_big_256:
	mpirun -np 8 -host marte,mercurio -npernode 4 ./sec ../input/pitufo.ppm ../output/pitufo-outtest_sec_big_256.ppm 20000 256
test_sec_small_512:
	mpirun -np 8 -host marte,mercurio -npernode 4 ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_big_512.ppm 7000 512
test_sec_big_512:
	mpirun -np 8 -host marte,mercurio  -npernode 4 ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_medium_512.ppm 20000 512
test_sec_small_2566:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_small_256.ppm 7000 256
test_sec_big_2566:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./sec ../input/pitufo.ppm ../output/pitufo-outtest_sec_big_256.ppm 20000 256
test_sec_small_5126:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_big_512.ppm 7000 512
test_sec_big_5126:
	mpirun -np 12 -host marte,mercurio  -npernode 6 ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_medium_512.ppm 20000 512
clean:
	rm -f sec
