import java.util.Scanner;
public class Soal7 {
    public static double sumMajorDiagonal(double[][] m){
        double hasil=0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if(i==j){
                    hasil += m[i][j];
                }
            }
        }
        return hasil;
    }
    public static void main(String[] args) {
        double[][] m = new double[4][4];
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] = sc.nextDouble();
            }
        }
        System.out.println("Sum of the elements in the major diagonal is " + sumMajorDiagonal(m));
    }
}
