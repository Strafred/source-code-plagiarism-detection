import java.util.Scanner;
public class fungsi {
    public static void main(String[] args) {
         System.out.print("Enter an integer: ");
        Scanner sc = new Scanner(System.in);
        int   input;
        input =    sc.nextInt();
        terbalik(input);
    }
    public static void terbalik(int angka)  {
        for(int i=0;i<=angka;i++) {
             int   sisa ;
             sisa= angka % 10;
            System.out.print(sisa);
            angka = angka / 10;
            i=0;
        }
        System.out.println();
    }
}
