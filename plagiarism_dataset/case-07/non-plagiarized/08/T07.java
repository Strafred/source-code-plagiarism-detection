import java.util.Scanner;
public class T07 {
    static Scanner sc = new Scanner(System.in);
    public static double sumMajorDiagonal(double[][] m) {
        double hasil = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] = sc.nextDouble();
                if(i==j){
                    hasil += m[i][j];
                }
            }
        }
        return hasil;
    }
    public static void main(String[] args) {
        double[][] m = new double[4][4];
        System.out.println("Enter a 4-by-4 matrix row by row:");
        System.out.println("Sum of the elements in the major diagonal is "+sumMajorDiagonal(m));
    }
}
