#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "../include/imagen.h"
#include "../include/ga.h"
#include "../include/derivados_mpi.h"

RGB *leer_ppm(const char *file, int *ancho, int *alto, int *max)
{
	int i, n;
	FILE *fd;

	char c, b[100];
	int red, green, blue;

	fd = fopen(file, "r");

	n = fscanf(fd, "%[^\n] ", b);
	if (b[0] != 'P' || b[1] != '3')
	{
		printf("%s no es una imagen PPM\n", file);
		exit(0);
	}

	printf("Leyendo fichero PPM %s\n", file);
	n = fscanf(fd, "%c", &c);
	while (c == '#')
	{
		n = fscanf(fd, "%[^\n] ", b);
		printf("%s\n", b);
		n = fscanf(fd, "%c", &c);
	}

	ungetc(c, fd);
	n = fscanf(fd, "%d %d %d", ancho, alto, max);
	assert(n == 3);

	int size = (*ancho) * (*alto);

	RGB *imagen = (RGB *)malloc(size * sizeof(RGB));
	assert(imagen);

	for (i = 0; i < size; i++)
	{
		n = fscanf(fd, "%d %d %d", &red, &green, &blue);
		assert(n == 3);

		imagen[i].r = red;
		imagen[i].g = green;
		imagen[i].b = blue;
	}

	fclose(fd);
	return imagen;
}

void escribir_ppm(const char *fichero, int ancho, int alto, int max, const RGB *imagen)
{
	int i;
	FILE *fd;

	fd = fopen(fichero, "w");

	fprintf(fd, "P3\n");
	fprintf(fd, "%d %d\n%d\n", ancho, alto, max);

	int size = alto * ancho;
	for (i = 0; i < size; i++)
	{
		fprintf(fd, "%d %d %d ", imagen[i].r, imagen[i].g, imagen[i].b);
		if ((i + 1) % 18 == 0)
		{
			fprintf(fd, "\n");
		}
	}
	fclose(fd);
}

void suavizar(int ancho, int alto, RGB *imagen)
{
	// Aplicar tecnica "mean-filter" para suavizar la imagen resultante
	int r = 0, g = 0, b = 0;
	//	(i-1, j-1) | (i-1, j) | (i-1, j+1)
	//	(i, j-1)   | p(i,j)   | (i, j+1)
	//	(i+1, j-1) | (i+1, j) | (i+1, j+1)
	int rank;
	int world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Datatype rgb_type;
	MPI_Datatype individuo_type;
	crear_tipo_datos(ancho * alto, &rgb_type, &individuo_type);
	int inicio = 0;
	if(rank == 0)
		inicio = MPI_Wtime();
	RGB *imagenLocal = (RGB *)malloc(ancho*(alto/world_size) * sizeof(RGB));
	int leftover = ancho*(alto%world_size);
	MPI_Scatter(imagen+leftover, ancho*(alto/world_size), rgb_type, imagenLocal,ancho*(alto/world_size), rgb_type, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		for (int i = 1; i < alto % world_size; i++)
		{
			//printf("i=%d, \n",i);

			for (int j = 1; j < ancho - 1; j++)
			{
				//printf("j=%d, ",j);

				r = imagen[(i - 1) * ancho + j - 1].r + imagen[(i - 1) * ancho + j].r + imagen[(i - 1) * ancho + j + 1].r +
					imagen[i * ancho + j - 1].r + imagen[i * ancho + j].r + imagen[i * ancho + j + 1].r +
					imagen[(i + 1) * ancho + j - 1].r + imagen[(i + 1) * ancho + j].r + imagen[(i + 1) * ancho + j + 1].r;

				g = imagen[(i - 1) * ancho + j - 1].g + imagen[(i - 1) * ancho + j].g + imagen[(i - 1) * ancho + j + 1].g +
					imagen[i * ancho + j - 1].g + imagen[i * ancho + j].g + imagen[i * ancho + j + 1].g +
					imagen[(i + 1) * ancho + j - 1].g + imagen[(i + 1) * ancho + j].g + imagen[(i + 1) * ancho + j + 1].g;

				b = imagen[(i - 1) * ancho + j - 1].b + imagen[(i - 1) * ancho + j].b + imagen[(i - 1) * ancho + j + 1].b +
					imagen[i * ancho + j - 1].b + imagen[i * ancho + j].b + imagen[i * ancho + j + 1].b +
					imagen[(i + 1) * ancho + j - 1].b + imagen[(i + 1) * ancho + j].b + imagen[(i + 1) * ancho + j + 1].b;

				imagen[i * ancho + j].r = round(r / 9);
				imagen[i * ancho + j].g = round(g / 9);
				imagen[i * ancho + j].b = round(b / 9);
			}
		}
		for (int j = 1; j < ancho - 1; j++) {
			imagen[j] = imagen[j + ancho];
			imagen[ancho*(alto%world_size-1)+j] = imagen[ancho*(alto%world_size-1)+ j - ancho];

		}
	}
	for (int i = 1; i < alto/world_size-1; i++)
	{
				//printf("i=%d, \n",i);

		for (int j = 1; j < ancho - 1; j++)
		{
				//printf("j=%d, ",j);

			r = imagenLocal[(i - 1) * ancho + j - 1].r + imagenLocal[(i - 1) * ancho + j].r + imagenLocal[(i - 1) * ancho + j + 1].r +
				imagenLocal[i * ancho + j - 1].r + imagenLocal[i * ancho + j].r + imagenLocal[i * ancho + j + 1].r +
				imagenLocal[(i + 1) * ancho + j - 1].r + imagenLocal[(i + 1) * ancho + j].r + imagenLocal[(i + 1) * ancho + j + 1].r;

			g = imagenLocal[(i - 1) * ancho + j - 1].g + imagenLocal[(i - 1) * ancho + j].g + imagenLocal[(i - 1) * ancho + j + 1].g +
				imagenLocal[i * ancho + j - 1].g + imagenLocal[i * ancho + j].g + imagenLocal[i * ancho + j + 1].g +
				imagenLocal[(i + 1) * ancho + j - 1].g + imagenLocal[(i + 1) * ancho + j].g + imagenLocal[(i + 1) * ancho + j + 1].g;

			b = imagenLocal[(i - 1) * ancho + j - 1].b + imagenLocal[(i - 1) * ancho + j].b + imagenLocal[(i - 1) * ancho + j + 1].b +
				imagenLocal[i * ancho + j - 1].b + imagenLocal[i * ancho + j].b + imagenLocal[i * ancho + j + 1].b +
				imagenLocal[(i + 1) * ancho + j - 1].b + imagenLocal[(i + 1) * ancho + j].b + imagenLocal[(i + 1) * ancho + j + 1].b;

			imagenLocal[i * ancho + j].r = round(r / 9);
			imagenLocal[i * ancho + j].g = round(g / 9);
			imagenLocal[i * ancho + j].b = round(b / 9);
		}
	}
		for (int j = 1; j < ancho - 1; j++) {
			imagenLocal[j] = imagenLocal[j + ancho];
			imagenLocal[ancho*(alto/world_size-1)+j] = imagenLocal[ancho*(alto/world_size-1)+ j - ancho];

		}
	MPI_Gather(imagenLocal,ancho*(alto/world_size), rgb_type, imagen+leftover, ancho*(alto/world_size), rgb_type, 0, MPI_COMM_WORLD); // Cambiar el tipo de MPI por el derivado

	if (rank == 0)
	{
		// Suavizar bordes
	

		imagen[0] = imagen[ancho + 1];
		imagen[ancho - 1] = imagen[ancho * 2 + 2];
		imagen[ancho * (alto - 1)] = imagen[ancho * (alto - 1) - ancho + 1];
		imagen[ancho * alto - 1] = imagen[ancho * alto - ancho - 2];


		/* Bordes izquierda y derecha */
		for (int i = 1; i < alto - 1; i++)
		{
			imagen[ancho * i] = imagen[ancho * i + 1];
			imagen[ancho * i - 1] = imagen[ancho * i - 2];
		}
		printf("Tiempo suavizar con MPI: %f\n",MPI_Wtime()-inicio);
	} 
}
