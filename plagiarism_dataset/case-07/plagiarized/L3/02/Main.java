import java.util.Scanner;
public class Main {
    public static double sumMajorDiagonal(double[][] mtx) {
        double sum = 0;
        for (int i = 0; i < mtx.length; i++)
            sum += mtx[i][i];
        return sum;
    }
    public static void main(String[] args) {
        double[][] mtx = new double[4][4];
        Scanner s = new Scanner(System.in);
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                mtx[i][j] = s.nextDouble();
        System.out.print("Sum of the elements in the major diagonal is "+ sumMajorDiagonal(mtx));
    }
}
