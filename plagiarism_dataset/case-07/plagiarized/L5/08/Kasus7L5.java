import java.util.Scanner;
public class Kasus7L5 {
    public static void main(String[] args) {
        Scanner inp = new Scanner(System.in);
        double[][] matrix = new double[4][4];
        int i = 0;
        double sum = 0;
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        while (i < 4) {
            int j = 0;
            while (j < 4) {
                matrix[i][j] = inp.nextDouble();
                j++;
            }
            i++;
        }
        i = 0;
        while (i < matrix.length) {
            sum += matrix[i][i];
            i++;
        }
        System.out.print("Sum of the elements in the major diagonal is " + sum);
    }
}
