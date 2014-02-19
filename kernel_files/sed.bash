#!/bin/bash

rm *.cl
cp ../ocl_kernels_bkup/* .

sed -i 's/'\
'if (row >= (y_min + 1)\s*\([+-]\?\s*[0-9]\?\)\s*'\
'&& row <= (y_max + 1)\s*\([+-]\?\s*[0-9]\?\)'\
'/IF_ROW_WITHIN(\1+0, \2+0)/g' *.cl

sed -i 's/'\
'&& column >= (x_min + 1)\s*\([+-]\?\s*[0-9]\?\)\s*'\
'&& column <= (x_max + 1)\s*\([+-]\?\s*[0-9]\?\))'\
'/IF_COLUMN_WITHIN(\1+0, \2+0)/g' *.cl

