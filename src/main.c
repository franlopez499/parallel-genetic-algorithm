#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../include/imagen.h"
#include "../include/ga.h"

static double mseconds() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1000 + t.tv_usec/1000;
}

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
	
	// Read Input Data
	RGB *imagen_objetivo = leer_ppm(argv[1], &ancho, &alto, &max);
	
	total = ancho*alto;
	// Allocate Memory for Output Data
	RGB *mejor_imagen = (RGB *) malloc(total*sizeof(RGB));

	#ifdef TIME
		double ti = mseconds();
	#endif
	
	// Call Genetic Algorithm
	crear_imagen(imagen_objetivo, total, ancho, alto, max, \
				 num_generaciones, tam_poblacion, mejor_imagen, argv[2]);	
	
	#ifdef TIME
		double tf = mseconds();
		printf("Execution Time = %.6lf seconds\n", (tf - ti));
	#endif
	
	// Smooth Output Image
	suavizar(ancho, alto, mejor_imagen);
	
	
	#ifdef DEBUG
		// Print Result
		escribir_ppm(argv[2], ancho, alto, max, mejor_imagen);
	#endif
	free(mejor_imagen);
	free(imagen_objetivo);
	
	return(0);
}
