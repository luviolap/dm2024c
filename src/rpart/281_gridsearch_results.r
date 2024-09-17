require(data.table)

dataset = fread("~/buckets/b1/exp/HT2810/gridsearch_detalle_0.txt")

table = dataset[,mean(ganancia_test),list(cp,minsplit,minbucket,maxdepth)]

fwrite(table, file="~/buckets/b1/exp/HT2810/agrupados_0.txt")