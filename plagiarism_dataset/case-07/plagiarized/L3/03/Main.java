import java.util.Scanner;
public class Main {
    public static double hasilPertambahanDiagonal(double[][] m) {
        double sum = 0;
        for (int i = 0; i < m.length; i++)
            sum += m[i][i];
        return sum;
    }
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        double[][] n = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                n[i][j] = input.nextDouble();
            }
        }
        System.out.print("Sum of the elements in the major diagonal is " + hasilPertambahanDiagonal(n));
    }
}
