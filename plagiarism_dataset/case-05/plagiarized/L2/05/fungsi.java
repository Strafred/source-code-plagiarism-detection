import java.util.Scanner;
public class fungsi {
    public static void main(String[] args) {
         System.out.print("Enter an integer: ");
        Scanner sc = new Scanner(System.in);
        int    input =    sc.nextInt();
          terbalik(input);
    }
    public static void terbalik(int angka)  {
        while (angka != 0) {
             int   sisa = angka % 10;
            System.out.print(sisa);
            angka = angka / 10;
        }
        System.out.println();
    }
}
