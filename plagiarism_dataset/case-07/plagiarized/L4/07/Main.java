import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        double[][] arr;
        Scanner input = new Scanner(System.in);
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        arr = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                arr[i][j] = input.nextDouble();
            }
        }
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i][i];
        }
        System.out.print("Sum of the elements in the major diagonal is "
                + sum);
    }
}
