Original C implementation of textile plot transformation


Requires LAPACK to run. For Ubuntu:

```sudo apt update
sudo apt install liblapack3
sudo apt install liblapack-dev 
sudo apt install libopenblas-base 
sudo apt install libopenblas-dev 
sudo apt install liblapacke-dev 
sudo apt install liblapack-dev
```
To compile

`gcc transform.c -lblas -llapack`

Expected output (Note: column 2,3,4 correspond to the input variables)

| 0.04888  | 0.08233  | 0.18339  | -0.11907 |
|----------|----------|----------|----------|
| __0.06249__  | __0.08162__  | __0.10585__  | __-0.11907__ |
| __0.23357__  | __0.18692__  | __0.03147__  | __0.48232__  |
| __-0.13687__ | __0.08246__  | __-0.29153__ | __-0.11907__ |
| __-0.04916__ | __0.03366__  | __-0.06206__ | __-0.11907__ |
| __-0.11256__ | __-0.21402__ | __0.03606__  | __-0.15971__ |
| __0.17796__  | __0.05154__  | __0.03661__  | __0.48232__  |
| __-0.03170__ | __0.11613__  | __-0.05152__ | __-0.15971__ |
| __0.01828__  | __0.12982__  | __0.04410__  | __-0.11907__ |
| __0.16077__  | __0.08246__  | __0.03661__  | __0.48232__  |
