import java.util.Scanner;
public class Main {
    static Scanner s = new Scanner(System.in);
    public static void inputData(double[][] mtx){
        int i =0;
        int j=0;
        while(i<4) {
            while(j<4){
                mtx[i][j] = s.nextDouble();
                j++;
            }
			j=0;
        i++;
        }
    }
    public static double sumMajorDiagonal(double[][] mtx) {
        double sum = 0;
        for (int i = 0; i < mtx.length; i++)
            sum += mtx[i][i];
        return sum;
    }
    public static void main(String[] args) {
        double[][] mtx = new double[4][4];
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        inputData(mtx);
        System.out.print("Sum of the elements in the major diagonal is "+ sumMajorDiagonal(mtx));
    }
}
