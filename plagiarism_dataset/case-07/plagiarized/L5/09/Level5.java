import java.util.Scanner;
public class Level5 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        double[][] matriks = new double[4][4];
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        for (int bar = 0; bar < 4; bar++) {
            for (int kol = 0; kol < 4; kol++) {
                matriks[bar][kol] = sc.nextDouble();
            }
        }
        hasil(matriks);
    }
    public static double sum(double[][] x) {
        double jumlah = 0;
        int i = 0;
        while(i < 4){
            jumlah += x[i][i];
            i++;
        }
        return jumlah;
    }
    private static void hasil(double[][] x) {
        System.out.print("Sum of the elements in the major diagonal is "+ sum(x));
    }
}
