#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "../include/imagen.h"
#include "../include/ga.h"


#define PRINT 0

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
	return (*(Individuo**)a)->fitness - (*(Individuo**)b)->fitness;
}



void crear_imagen(const RGB *imagen_objetivo, int num_pixels, int ancho, int alto, int max, int num_generaciones, int tam_poblacion, RGB *imagen_resultado, const char *output_file)
{
	double initial_time_fitness = 0;
	double final_time_fitness = 0;
	double total_time_fitness = 0;
	
	double initial_time_cruzar = 0;
	double final_time_cruzar = 0;
	double total_time_cruzar = 0;
	
	double initial_time_sort = 0;
	double final_time_sort = 0;
	double total_time_sort = 0;


	double initial_time_mutar = 0;
	double final_time_mutar = 0;
	double total_time_mutar = 0;

	int i, mutation_start;
//	char output_file2[32];
	double fitness_anterior, fitness_actual, diferencia_fitness;
	
	// A. Crear Poblacion Inicial (array de imagenes aleatorias)
	Individuo **poblacion = (Individuo **) malloc(tam_poblacion*sizeof(Individuo));
	assert(poblacion);
	
	for(i = 0; i < tam_poblacion; i++) {
		poblacion[i] = (Individuo *) malloc(sizeof(Individuo));
		poblacion[i]->imagen = imagen_aleatoria(max, num_pixels);
	}
	
	initial_time_fitness = omp_get_wtime();
	#pragma omp parallel for schedule(guided)
	for (i = 0; i < tam_poblacion; i++) {
		fitness(imagen_objetivo, poblacion[i], num_pixels);
	}
	final_time_fitness = omp_get_wtime();
	total_time_fitness+= final_time_fitness - initial_time_fitness;
	
	// Ordenar individuos según la función de bondad (menor "fitness" --> más aptos)
	qsort(poblacion, tam_poblacion, sizeof(Individuo *), comp_fitness);



	// B. Evolucionar la Población (durante un número de generaciones)
	for(int g = 0; g < num_generaciones; g++) {
		fitness_anterior = poblacion[0]->fitness;
		initial_time_cruzar = omp_get_wtime();
		// Promocionar a los descendientes de los individuos más aptos
		#pragma omp parallel for shared(poblacion) schedule(guided)
		for(i = 0; i < (tam_poblacion/2); i += 2) {
			cruzar(poblacion[i], poblacion[i+1], poblacion[tam_poblacion/2+i], poblacion[tam_poblacion/2+i+1], num_pixels);
		}
		final_time_cruzar  = omp_get_wtime();
		total_time_cruzar += final_time_cruzar - initial_time_cruzar;
		// Mutar una parte de la individuos de la población (se decide que muten tam_poblacion/4)
		mutation_start = tam_poblacion/4;
		initial_time_mutar = omp_get_wtime();
		for(i = mutation_start; i < tam_poblacion; i++) {
			mutar(poblacion[i], max, num_pixels);
		}
		final_time_mutar = omp_get_wtime();
		total_time_mutar += final_time_mutar - initial_time_mutar;

		initial_time_fitness = omp_get_wtime();
		// Recalcular Fitness
		#pragma omp parallel for schedule(guided)
		for(i = 0; i < tam_poblacion; i++) {
			fitness(imagen_objetivo, poblacion[i], num_pixels);
		}
		final_time_fitness = omp_get_wtime();
		total_time_fitness += final_time_fitness - initial_time_fitness;
		// Ordenar individuos según la función de bondad (menor "fitness" --> más aptos)
		initial_time_sort = omp_get_wtime();
		
			// Ordenar individuos según la función de bondad (menor "fitness" --> más aptos)
		qsort(poblacion, tam_poblacion, sizeof(Individuo *), comp_fitness);
		final_time_sort  = omp_get_wtime();
		total_time_sort += final_time_sort - initial_time_sort;

		
		// La mejor solución está en la primera posición del array
		fitness_actual = poblacion[0]->fitness;
		diferencia_fitness = -(fitness_actual-fitness_anterior)/fitness_actual*100;
		
		// Guardar cada 300 iteraciones para observar el progreso
		if (PRINT) {
			printf("Generacion %d - ", g);
			printf("Fitness = %e - ", fitness_actual);
			printf("Diferencia con Fitness Anterior = %.2e%c\n", diferencia_fitness, 37);
//			if ((g % 300) == 0) {
//				printf("%s\n",output_file);
//				sprintf(output_file2,"image_%d.ppm",g);
//				escribir_ppm(output_file2, ancho, alto, max, poblacion[0]->imagen);
//			}
		}
	}
	
	printf("Tiempo invertido en cálculo cruzar: %f\n", total_time_cruzar);
	printf("Tiempo invertido en cálculo sort: %f\n", total_time_sort);
	printf("Tiempo invertido en cálculo mutar: %f\n", total_time_mutar);
	printf("Tiempo invertido en cálculo fitness: %f\n", total_time_fitness);

	// Devuelve Imagen Resultante
	memmove(imagen_resultado, poblacion[0]->imagen, num_pixels*sizeof(RGB));
	
	// Release memory
	for(i = 0; i < tam_poblacion; i++) {
		free(poblacion[i]->imagen);
		free(poblacion[i]);
	}
	free(poblacion);
}

void cruzar(Individuo *padre1, Individuo *padre2, Individuo *hijo1, Individuo *hijo2, int num_pixels)
{
	// Elegir un "punto" de corte aleatorio a partir del cual se realiza el intercambio de los genes.
	// * Cruzar los genes de cada padre con su hijo
	// * Intercambiar los genes de cada hijo con los del otro padre
	int corte = aleatorio(num_pixels - 1);
		#pragma omp parallel 
		{
			
			#pragma omp for nowait
            for(int i = 0; i < corte; i++) {
		    	hijo1->imagen[i] = padre1->imagen[i];
		     	hijo2->imagen[i] = padre2->imagen[i];
        	}
        	
			#pragma omp for 
            for(int j = corte; j < num_pixels; j++) {
		        hijo1->imagen[j] = padre2->imagen[j];
		        hijo2->imagen[j] = padre1->imagen[j];
			}
		}
}

void fitness(const RGB *objetivo, Individuo *individuo, int num_pixels)
{
	// Determina la calidad del individuo (similitud con el objetivo)
	// calculando la suma de la distancia existente entre los pixeles
	double fitness = 0;
	#pragma omp parallel for reduction(+:fitness)
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

    #pragma omp parallel for schedule(guided) if(numMutar > num_pixels/3)    
	for(int i = 0; i < numMutar; i++) {
		int index = aleatorio(num_pixels - 1);

		actual->imagen[index].r = aleatorio(max);
		actual->imagen[index].g = aleatorio(max);
		actual->imagen[index].b = aleatorio(max);
	}

}
