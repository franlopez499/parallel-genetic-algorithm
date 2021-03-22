CC = mpicc
CFLAGS = -O3 -std=c99 -g -Wall

mpi: main.c imagen.c ga.c derivados_mpi.c
	$(CC) $(CFLAGS) main.c imagen.c ga.c derivados_mpi.c -o mpi -DTIME -DDEBUG -lm

test_mpi_sm_7:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./mpi ../input/pitufo.ppm ../output/t12/pitufoOut_sm_7.ppm 7000 256

test_mpi_bg_7:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./mpi ../input/pitufo.ppm ../output/t12/pitufoOut_bg_7.ppm 7000 512

test_mpi_sm_20:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./mpi ../input/pitufo.ppm ../output/t12/pitufoOut_sm_20.ppm 20000 256

test_mpi_bg_20:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./mpi ../input/pitufo.ppm ../output/t12/pitufoOut_bg_20.ppm 20000 512

test_mpi_dn_bg:
	mpirun -np 12 -host marte,mercurio -npernode 6 ./mpi ../input/pitufo.ppm ../output/dn33/pitufoOut_bg5.ppm 5000 256

clean:
	rm -f mpi
