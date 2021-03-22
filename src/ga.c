#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "../include/derivados_mpi.h"
#include "../include/imagen.h"
#include "../include/ga.h"

#define PRINT 0

extern int rank, world_size;
extern MPI_Datatype rgb_type, individuo_type;
int buff_size;
Individuo *packBuff;


static int aleatorio(int max) {
	return (rand() % (max+1));
}

void init_imagen_aleatoria(RGB *imagen, int max, int total)
{
	for(int i = 0; i < total; i++) {
		imagen[i].r = aleatorio(max);
		imagen[i].g = aleatorio(max);
		imagen[i].b = aleatorio(max);
	}
}

RGB *imagen_aleatoria(int max, int total) {
	RGB *imagen = (RGB *) malloc(total*sizeof(RGB));
	assert(imagen);
	
	init_imagen_aleatoria(imagen, max, total);
	return imagen;
}

static int comp_fitness(const void *a, const void *b) {
	/* qsort pasa un puntero al elemento que está ordenando */
	return (*(Individuo*)a).fitness - (*(Individuo*)b).fitness;
}

void scatterPoblacion(int *offset, int chunk_size, int leftover, Individuo *pobMaster, Individuo *pobRecv)
{
  MPI_Status status;
  int tagPack = 0;
  int position;
  if(rank == MASTER) {

   *offset += chunk_size + leftover;
   for(int i = 1; i < world_size; i++) {
     position = 0;
     MPI_Pack(offset, 1, MPI_INT, packBuff, buff_size, &position, MPI_COMM_WORLD); 
     MPI_Pack(&(pobMaster[*offset]), chunk_size, individuo_type, packBuff, buff_size, &position, MPI_COMM_WORLD);
     MPI_Send(packBuff, position, MPI_PACKED, i, tagPack, MPI_COMM_WORLD);
     *offset += chunk_size;
   }
  } else {
    position = 0;
    MPI_Recv(packBuff, buff_size, MPI_PACKED, MASTER, tagPack, MPI_COMM_WORLD, &status);
    MPI_Unpack(packBuff, buff_size, &position, offset, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(packBuff, buff_size, &position, pobRecv, chunk_size, individuo_type, MPI_COMM_WORLD);
  }
}

void gatherPoblacion(int *offset, int chunk_size, Individuo *pobMaster, Individuo *pobSend)
{
  MPI_Status status;
  int tagPack = 0;
  int position;

  if(rank == MASTER) {
    for(int i = 1; i < world_size; i++) {
      position = 0;
      MPI_Recv(packBuff, buff_size, MPI_PACKED, MPI_ANY_SOURCE, tagPack, MPI_COMM_WORLD, &status);
      MPI_Unpack(packBuff, buff_size, &position, offset, 1, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack(packBuff, buff_size, &position, &(pobMaster[*offset]), chunk_size, individuo_type, MPI_COMM_WORLD);
    }
  } else {
      position = 0;
      MPI_Pack(offset, 1, MPI_INT, packBuff, buff_size, &position, MPI_COMM_WORLD);
      MPI_Pack(pobSend, chunk_size, individuo_type, packBuff, buff_size, &position, MPI_COMM_WORLD);
      MPI_Send(packBuff, position, MPI_PACKED, MASTER, tagPack, MPI_COMM_WORLD);
  }
}

void crear_imagen(const RGB *imagen_objetivo, int num_pixels, int ancho, int alto, int max, int num_generaciones, int tam_poblacion, RGB *imagen_resultado, const char *output_file)
{
  int i, mutation_start;
  double fitness_anterior, fitness_actual, diferencia_fitness;
  int chunk_size = tam_poblacion / world_size;
  int leftover = tam_poblacion % world_size;
  int offset = 0;

  // Se reserva memoria para el empaquetado de datos
  buff_size = sizeof(int) + sizeof(Individuo) * chunk_size;
  packBuff = (Individuo *) malloc(buff_size);

  // Se crea la población inicial
  Individuo *poblacion = NULL;
  if(rank == MASTER) {
    poblacion = (Individuo *) malloc(tam_poblacion * sizeof(Individuo));
    for(i = 0; i < tam_poblacion; i++) {
      init_imagen_aleatoria(poblacion[i].imagen, max, num_pixels);
    }  
  } else {
    poblacion = (Individuo *) malloc(chunk_size * sizeof(Individuo));
  }
  
  // Calcular fitness
  if(rank == MASTER) {
    offset = 0;
    scatterPoblacion(&offset, chunk_size, leftover, poblacion, NULL);
    chunk_size += leftover;
  } else {
    scatterPoblacion(&offset, chunk_size, leftover, NULL, poblacion);
  }
  for(i = 0; i < chunk_size; i++) {
    fitness(imagen_objetivo, &poblacion[i], num_pixels);
  }

  if(rank == MASTER) {
    chunk_size = tam_poblacion / world_size;
    gatherPoblacion(&offset, chunk_size, poblacion, NULL);
    // Se ordenan los individuos según su fitness
    qsort(poblacion, tam_poblacion, sizeof(Individuo), comp_fitness);

    offset = 0;
    scatterPoblacion(&offset, chunk_size, leftover, poblacion, NULL);
    chunk_size += leftover;
  } else {
    gatherPoblacion(&offset, chunk_size, NULL, poblacion);
    scatterPoblacion(&offset, chunk_size, leftover, NULL, poblacion);
  }

  // Evolucionar la población durante un número de generaciones
  for(int g = 0; g < num_generaciones; g++) {
    if(rank == MASTER) {
      fitness_anterior = poblacion[0].fitness;
    }

    for(i = 0; i < chunk_size/2 - 1; i += 2) {
      cruzar(&poblacion[i], &poblacion[i+1], &poblacion[chunk_size/2+i], &poblacion[chunk_size/2+i+1], num_pixels);
    }

    // Mutar una parte de la población
    mutation_start = chunk_size/4;

    for(i = mutation_start; i < chunk_size; i++) {
      mutar(&poblacion[i], max, num_pixels);
    }
    
    // Recalcular fitness
      for(i = 0; i < chunk_size; i++) {
        fitness(imagen_objetivo, &poblacion[i], num_pixels);
      }
      
      if((g % 10 == 0 && g) || g == num_generaciones -1){
        if(rank == MASTER) {
          chunk_size = tam_poblacion / world_size;
          gatherPoblacion(&offset, chunk_size, poblacion, NULL);
          // Se ordenan los individuos según su fitness
          qsort(poblacion, tam_poblacion, sizeof(Individuo), comp_fitness);
          
          // La mejor solución está en la primera posición del array
          fitness_actual = poblacion[0].fitness;
          diferencia_fitness = -(fitness_actual-fitness_anterior)/fitness_actual*100;

          // Guardar cada 300 iteraciones para observar el progreso
          if (PRINT) {
            printf("Generacion %d - ", g);
            printf("Fitness = %e - ", fitness_actual);
            printf("Diferencia con Fitness Anterior = %.2e%c\n", diferencia_fitness, 37);
            // if ((g % 300) == 0) {
            // printf("%s\n",output_file);
            // sprintf(output_file2,"image_%d.ppm",g);
            // escribir_ppm(output_file2, ancho, alto, max, poblacion[0]->imagen);
            // }
          }

          offset = 0;
          scatterPoblacion(&offset, chunk_size, leftover, poblacion, NULL);
          chunk_size += leftover;
        } else {
            gatherPoblacion(&offset, chunk_size, NULL, poblacion);
            scatterPoblacion(&offset, chunk_size, leftover, NULL, poblacion);
        }
      } 
        qsort(poblacion, chunk_size, sizeof(Individuo), comp_fitness);
  }
  if(rank == MASTER) {
    // Devuelve la imagen resultante
    memmove(imagen_resultado, poblacion[0].imagen, num_pixels*sizeof(RGB));
  } 
  free(packBuff);
  free(poblacion);
}


void cruzar(Individuo *padre1, Individuo *padre2, Individuo *hijo1, Individuo *hijo2, int num_pixels)
{
	// Elegir un "punto" de corte aleatorio a partir del cual se realiza el intercambio de los genes.
	// * Cruzar los genes de cada padre con su hijo
	// * Intercambiar los genes de cada hijo con los del otro padre
	int corte = aleatorio(num_pixels - 1);

	for(int i = 0; i < corte; i++) {
		hijo1->imagen[i] = padre1->imagen[i];
		hijo2->imagen[i] = padre2->imagen[i];
	}

	for(int i = corte; i < num_pixels; i++) {
		hijo1->imagen[i] = padre2->imagen[i];
		hijo2->imagen[i] = padre1->imagen[i];
	}

}

void fitness(const RGB *objetivo, Individuo *individuo, int num_pixels)
{
	// Determina la calidad del individuo (similitud con el objetivo)
	// calculando la suma de la distancia existente entre los pixeles
  /* if(rank != MASTER) { */
  /*   printf("Fitness - %d\n", rank); */
  /*   printf("Pixel: %d\n", objetivo[1].r); */
  /*   printf("Ind: %d\n", individuo->imagen[1].r); */
  /* } */
	double fitness = 0;
	for(int i = 0; i < num_pixels; i++) {

		fitness += abs(objetivo[i].r - individuo->imagen[i].r) + abs(objetivo[i].g - individuo->imagen[i].g) + abs(objetivo[i].b - individuo->imagen[i].b);
	}

	individuo->fitness = fitness;
}

void mutar(Individuo *actual, int max, int num_pixels)
{
	// Cambia el valor de algunos puntos de la imagen de forma aleatoria.
	
	// Decidir cuantos pixels mutar. Si el valor es demasiado pequeño,
	// la convergencia es muy pequeña, y si es demasiado alto diverge.
	double ratioMutacion = 0.002;
	int numMutar = (int) num_pixels * ratioMutacion;

	for(int i = 0; i < numMutar; i++) {
		int index = aleatorio(num_pixels - 1);

		actual->imagen[index].r = aleatorio(max);
		actual->imagen[index].g = aleatorio(max);
		actual->imagen[index].b = aleatorio(max);
	}

}
