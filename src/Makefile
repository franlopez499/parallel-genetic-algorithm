CC = mpicc
CFLAGS = -O3 -std=c99 -g -Wall

sec: main.c imagen.c ga.c derivados_mpi.c
	$(CC) $(CFLAGS) main.c imagen.c ga.c derivados_mpi.c -o sec -DTIME -DDEBUG -lm
test_sec_small_256:
	mpirun -np 6 -host marte ./sec ../input/pitufo.ppm ../output/pitufo-out_test_sec_small_256.ppm 7000 256
test_sec_big_256:
	mpirun -np 4 -host marte ./sec ../input/pitufo.ppm ../output/pitufo-outtest_sec_big_256.ppm 30000 256
clean:
	rm -f sec

