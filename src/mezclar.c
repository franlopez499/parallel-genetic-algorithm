#include <stdlib.h>
#include <omp.h>

#include "../include/ga.h"

void mezclar(Individuo **x, int izq, int med, int der)
{
	int i, j, k;
	Individuo **temp = (Individuo **) malloc(sizeof(Individuo)*(der+1));
	
	k = 0;
	i = izq;
	j = med;
	while(i<med && j<der) {
		if(x[i]->fitness < x[j]->fitness) {
			temp[k] = x[i];
			i++;
		}
		else {
			temp[k] = x[j];
			j++;
		}
		k++;
	}
	
	for(; i<med; i++) {
		temp[k] = x[i];
		k++;
	}
	
	for(; j<der; j++) {
		temp[k] = x[j];
		k++;
	}
	
	for(i=0; i<der-izq; i++) {
		x[i+izq] = temp[i];
	}
	free(temp);
}

void mergeSort(Individuo **poblacion, int tam)
{
	if (tam < 2) return;

	#pragma omp task if(tam > 75)
	mergeSort(poblacion, tam/2);
	#pragma omp task if(tam > 75)
	mergeSort(&poblacion[tam/2], tam/2);

	#pragma omp taskwait
	mezclar(poblacion, 0, tam/2, tam);
}
