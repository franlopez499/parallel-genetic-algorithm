#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include "../include/imagen.h"
#include "../include/ga.h"
#include "../include/derivados_mpi.h"

/* static double mseconds() { */
/* 	struct timeval t; */
/* 	gettimeofday(&t, NULL); */
/* 	return t.tv_sec*1000 + t.tv_usec/1000; */
/* } */

int rank, world_size;
MPI_Datatype rgb_type, individuo_type;

int main(int argc, char **argv)
{
	int ancho, alto, max, total;
	int num_generaciones, tam_poblacion;
	
	ancho = alto = max = 0;
	num_generaciones = tam_poblacion = 0;
	
	// Check Input
	if(argc < 4) {
		printf("Ayuda:\n"); 
		printf("  ./programa entrada salida num_generaciones tam_poblacion\n");
		return(-1);
	}
	num_generaciones = atoi(argv[3]);
	tam_poblacion = atoi(argv[4]);
	
	if (tam_poblacion % 4 != 0) {
		printf("El tamaño de la población debe ser divisible por 4\n");
		return(-1);
	}

  double wtime = 0;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  RGB *imagen_objetivo = NULL;
  RGB *mejor_imagen = NULL;
  
   if(rank == MASTER) {
	// Read Input Data
    imagen_objetivo = leer_ppm(argv[1], &ancho, &alto, &max);
    
    total = ancho*alto;
    crear_tipo_datos(total, &rgb_type, &individuo_type);

    for(int i = 1; i < world_size; i++) {
      MPI_Send(&ancho, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&alto, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
      MPI_Send(&max, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
      MPI_Send(imagen_objetivo, total, rgb_type, i, 3, MPI_COMM_WORLD);
    }
    // Allocate Memory for Output Data
  } else {
    MPI_Recv(&ancho, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&alto, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&max, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD, &status);
    total = ancho * alto;
    crear_tipo_datos(total, &rgb_type, &individuo_type);
    imagen_objetivo = (RGB *) malloc(total*sizeof(RGB));
    MPI_Recv(imagen_objetivo, total, rgb_type, MASTER, 3, MPI_COMM_WORLD, &status);
  }
	// #ifdef TIME
	// 	double ti = mseconds();
	// #endif

  
  if(rank == MASTER) {
    mejor_imagen = (RGB *) malloc(total*sizeof(RGB));
    wtime = MPI_Wtime();
  }
	// Call Genetic Algorithm
	crear_imagen(imagen_objetivo, total, ancho, alto, max, \
				 num_generaciones, tam_poblacion, mejor_imagen, argv[2]);	

  if(rank == MASTER) {
    wtime = MPI_Wtime() - wtime;
 		printf("Execution Time = %.6lf seconds\n", wtime); 
  }
	
	// Smooth Output Image
	suavizar(ancho, alto, mejor_imagen);
	
	
	#ifdef DEBUG
		// Print Result
    if(rank == MASTER) {
      escribir_ppm(argv[2], ancho, alto, max, mejor_imagen);
    }
	#endif
	free(mejor_imagen);
	free(imagen_objetivo);
  MPI_Finalize();	
	return(0);
}
