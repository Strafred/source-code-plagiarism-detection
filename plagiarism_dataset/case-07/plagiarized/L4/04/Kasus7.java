import java.util.Scanner;
public class Kasus7 {
    public static void inputMatrix(double[][] matrix, int n) {
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = sc.nextDouble();
            }
        }
    }
    public static void sumMajorDiagonal(double[][] matrix) {
        double sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            sum += matrix[i][i];
        }
        System.out.println("Sum of the elements in the major diagonal is " + sum);
    }
    public static void main(String[] args) {
        int n=4;
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        double[][] matrix = new double[n][n];
        inputMatrix(matrix, n);
        sumMajorDiagonal(matrix);
    }
}
