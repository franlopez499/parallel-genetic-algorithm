#ifndef _GA
#define _GA

#include "imagen.h"

typedef struct {
	RGB *imagen;
	double fitness;
} Individuo;

void crear_imagen(const RGB *, int, int, int, int, int, int, RGB *, const char *);
void cruzar(Individuo *, Individuo *, Individuo *, Individuo *, int);
void fitness(const RGB *, Individuo *, int);
void mutar(Individuo *, int, int);

#endif
