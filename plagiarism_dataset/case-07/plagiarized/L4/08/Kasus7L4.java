import java.util.Scanner;
public class Kasus7L4 {
    public static void main(String[] args) {
        Scanner inp = new Scanner(System.in);
        double[][] matrix = new double[4][4];
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                matrix[i][j] = inp.nextDouble();
            }
        }
        double sum = 0;
        for (int i = 0; i < matrix.length; i++) {
            sum += matrix[i][i];
        }
        System.out.print("Sum of the elements in the major diagonal is " + sum);
    }
}
