#ifndef _GA
#define _GA

#include "imagen.h"

typedef struct {
	RGB imagen[11904]; // 11904 cambiar a 128*128 para otros ppm tras ejecutar el conversor
	double fitness;
} Individuo;

void crear_imagen(const RGB *, int, int, int, int, int, int, RGB *, const char *);
void cruzar(Individuo *, Individuo *, Individuo *, Individuo *, int);
void fitness(const RGB *, Individuo *, int);
void mutar(Individuo *, int, int);

#endif
