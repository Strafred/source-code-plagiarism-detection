import java.util.Scanner;
public class Level3 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        double[][] matriks = new double[4][4];
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        for (int bar = 0; bar < 4; bar++) {
            for (int kol = 0; kol < 4; kol++) {
                matriks[bar][kol] = scan.nextDouble();
            }
        }
        System.out.print("Sum of the elements in the major diagonal is "
                + diagonal(matriks));
    }
    public static double diagonal(double[][] x) {
        double jumlah = 0;
        for (int i = 0; i < x.length; i++)
            jumlah += x[i][i];
        return jumlah;
    }
}
