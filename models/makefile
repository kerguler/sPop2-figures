model: model.c modelB.c modelC.c
	gcc -L/usr/local/lib -I/usr/local/include -Wall -lm -lspop2 -lgsl -shared -fPIC -single_module -o model.dylib model.c
	gcc -L/usr/local/lib -I/usr/local/include -Wall -lm -lspop2 -lgsl -shared -fPIC -single_module -o modelB.dylib modelB.c
	gcc -L/usr/local/lib -I/usr/local/include -Wall -lm -lspop2 -lgsl -shared -fPIC -single_module -o modelC.dylib modelC.c