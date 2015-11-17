
all : 201x_acc_fraction

201x_acc_fraction : TOP500_201011.csv  TOP500_201111.csv  TOP500_201206.csv  TOP500_201211.csv  TOP500_201306.csv  TOP500_201311.csv  TOP500_201406.csv  TOP500_201411.csv  TOP500_201506.csv  TOP500_201511.csv
	./plot.R $^

clean : 201x_acc_fraction.png 201x_acc_fraction.svg
	rm -f $^
