import java.util.Scanner;
public class T06 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] array = new int[10];
        for (int i = 0; i < array.length; i++) {
            System.out.print("Read a number: ");
            array[i] = sc.nextInt();
        }
        System.out.println(array[9]);
        System.out.println(array[8]);
        System.out.println(array[7]);
        System.out.println(array[6]);
        System.out.println(array[5]);
        System.out.println(array[4]);
        System.out.println(array[3]);
        System.out.println(array[2]);
        System.out.println(array[1]);
        System.out.println(array[0]);
    }
}
