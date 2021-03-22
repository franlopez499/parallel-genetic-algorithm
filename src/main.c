#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "../include/imagen.h"
#include "../include/ga.h"
#include "../include/derivados_mpi.h"

static double mseconds()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec * 1000 + t.tv_usec / 1000;
}

void escribeArray(RGB *a, int total)
{
	for (int i = 0; i < total; i++)
	{
		printf("%d ", a->g);
	}
}
int main(int argc, char **argv)
{

	int ancho, alto, max, total;
	int num_generaciones, tam_poblacion;
	int rank, world_size;

	ancho = alto = max = 0;
	num_generaciones = tam_poblacion = 0;

	// Check Input
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	RGB *imagen_objetivo = NULL; 
	if (rank == 0)
	{
		if (argc < 4)
		{
			printf("Ayuda:\n");
			printf("  ./programa entrada salida num_generaciones tam_poblacion\n");
			return (-1);
		}
		num_generaciones = atoi(argv[3]);
		tam_poblacion = atoi(argv[4]);

		if (tam_poblacion % 4 != 0)
		{
			printf("El tamaño de la población debe ser divisible por 4\n");
			return (-1);
		}

		// Read Input Data

		imagen_objetivo = leer_ppm(argv[1], &ancho, &alto, &max);
		printf("leida objetivo");

		total = ancho * alto;
		// Allocate Memory for Output Data
	}
	MPI_Datatype rgb_type;
	MPI_Datatype individuo_type;
	crear_tipo_datos(ancho * alto, &rgb_type, &individuo_type);

	MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ancho, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alto, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&max, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&num_generaciones, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&tam_poblacion, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(rank != 0){
		imagen_objetivo = malloc(total*sizeof(RGB));
	MPI_Bcast(imagen_objetivo, total, rgb_type, 0, MPI_COMM_WORLD);

	}else{
	MPI_Bcast(imagen_objetivo, total, rgb_type, 0, MPI_COMM_WORLD);

	}
	RGB *mejor_imagen = (RGB *)malloc(total * sizeof(RGB));
	

#ifdef TIME
	double ti = mseconds();
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	// Call Genetic Algorithm
	crear_imagen(imagen_objetivo, total, ancho, alto, max,
				 num_generaciones, tam_poblacion, mejor_imagen, argv[2]);

#ifdef TIME
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
			double tf = mseconds();
		printf("Execution Time = %.6lf seconds\n", (tf - ti));
	}

#endif
	MPI_Barrier(MPI_COMM_WORLD);

	// Smooth Output Image
	suavizar(ancho, alto, mejor_imagen);

#ifdef DEBUG
	// Print Result
	if(rank == 0)
		escribir_ppm(argv[2], ancho, alto, max, mejor_imagen);
#endif
		MPI_Barrier(MPI_COMM_WORLD);

	free(mejor_imagen);
	free(imagen_objetivo);
	MPI_Finalize();

	return (0);
}
