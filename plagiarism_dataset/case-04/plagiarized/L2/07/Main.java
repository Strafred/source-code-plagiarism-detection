public class Main {
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        int jarak = 1;
        while (jarak <= 10) {
            System.out.println(jarak + "\t\t" + jarak * 1.609);
            jarak++;
        }
    }
}
